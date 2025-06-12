from typing import Optional, Tuple, List
import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from transformers.modeling_utils import PreTrainedModel
from transformers.models.llama.modeling_llama import (
    apply_rotary_pos_emb,
    repeat_kv,
)
from transformers.cache_utils import DynamicCache

from pylibraft.cluster import KMeansParams, fit
from pylibraft.neighbors import ivf_flat
import pylibraft.config
pylibraft.config.set_output_as("torch")
import rmm
# torch.set_printoptions(profile="full")

from clusterkv._clusterkv_knl import search_indices
from .cluster_cache_simulator import CacheSimulator

# Use this function as the metadata only has 2-dim
def repeat_metadata(metadata: torch.Tensor, n_rep: int) -> torch.Tensor:
    num_key_value_heads, slen = metadata.shape
    if n_rep == 1:
        return metadata
    metadata = metadata[:, None, :].expand(num_key_value_heads, n_rep, slen)
    return metadata.reshape(num_key_value_heads * n_rep, slen)

def build_cluster(prefill_key, nlist, balance, cluster_params, 
                  num_key_value_groups, gqa_policy):
    _, num_kv_heads, prefill_len, head_dim = prefill_key.shape
    nlist_range = torch.arange(nlist, device=prefill_key.device).reshape(nlist, 1)
    cluster_key_indices = torch.empty((num_kv_heads, prefill_len), dtype=torch.int64,
                                            device=prefill_key.device)
    cluster_key_ptr = torch.empty((num_kv_heads, prefill_len), dtype=torch.int16,
                                        device=prefill_key.device)
    cluster_key_size = torch.empty((num_kv_heads, nlist), dtype=torch.int32,
                                        device=prefill_key.device)
    for h in range(num_kv_heads):
        head_keys = prefill_key[0, h].to(torch.float32)
        if balance:
            flat_index = ivf_flat.build(cluster_params, head_keys)
            head_centroids = flat_index.centers
        else:
            head_centroids, _, _ = fit(cluster_params, head_keys)
        head_centroids = head_centroids.to(prefill_key.dtype)
        # centoid_indices: (prefill_len,)
        _, centoid_indices = torch.max(torch.mm(F.normalize(prefill_key[0, h], p=2, dim=-1), 
                                                F.normalize(head_centroids, p=2, dim=-1).t()), 
                                                dim=-1)
        # if centoid_indices is like [3, 1, 1, 2]
        # cluster_key_ptr is [1, 1, 2, 3], cluster_key_indices is [1, 2, 3, 0]
        cluster_key_ptr[h], cluster_key_indices[h] \
            = torch.where(centoid_indices==nlist_range)
        cluster_key_size[h] = torch.bincount(cluster_key_ptr[h], minlength=nlist)
        # self.cluster_key[0, h] = prefill_key[0, h, self.cluster_key_indices[h], :]
        
        # if self.layer_id == 10 and h == 1:
        #     print(centoid_indices)
        #     print(self.cluster_key_indices[h], self.cluster_key_ptr[h])
        #     print(self.cluster_key_size[h])
        head_centroids = head_centroids.unsqueeze(0)
        if h == 0:
            key_centroids = head_centroids
        else:
            key_centroids = torch.cat([key_centroids, head_centroids], dim=0)
    # (num_kv_heads, nlist)
    cluster_key_size_ps = torch.cumsum(cluster_key_size, dim=-1)
    # if self.layer_id == 10:
    #     print(self.cluster_key_size_ps[1])
    key_centroids = key_centroids.unsqueeze(0)
    # self.key_centroids: (1, num_kv_heads, nlist, head_dim)
    if gqa_policy is None:
        key_centroids = repeat_kv(key_centroids, num_key_value_groups)
        cluster_key_ptr = repeat_metadata(cluster_key_ptr, num_key_value_groups)
        cluster_key_size = repeat_metadata(cluster_key_size, num_key_value_groups)
        cluster_key_size_ps = repeat_metadata(cluster_key_size_ps, num_key_value_groups)

    return key_centroids, cluster_key_indices, cluster_key_ptr, cluster_key_size, cluster_key_size_ps

def stat_topk(layer_id, indices, q, prefill_keys, name):
    _, num_heads, k = indices.shape
    attn_weights = torch.matmul(q, prefill_keys.transpose(2, 3))    # [1, num_heads, 1, seq_len]
    _, topk_indices = attn_weights.topk(k, dim=-1)
    topk_indices = topk_indices.squeeze(2)      # [1, 32, k]
    hit_rate = []
    for h in range(num_heads):
        truth = topk_indices[0, h].cpu()
        pred = indices[0, h].cpu()
        hit_rate.append(len( set(truth.numpy()) & set(pred.numpy()) ) / k)
    avg_hit_rate = sum(hit_rate) / len(hit_rate)
    with open(f'topk_stat/top{k}-{name}.csv', 'a') as f:
        f.write(f'layer {layer_id}, {avg_hit_rate}\n')

def cluster_attn_out(query_states, key_states, value_states, attention_mask, prompt_len,
                    key_centroids, cluster_key_indices, cluster_key_size, cluster_key_size_ps,
                    num_key_value_groups, layer_id, token_budget, sink, head_sel, 
                    cluster_cache, topk_stat=False, cluster_params=None):
    bsz, num_kv_heads, kv_seq_len, head_dim = key_states.shape
    _, num_heads, q_len, _ = query_states.shape
    hidden_size = num_heads * head_dim
	# c_dist: (1, num_heads, 1, nlist)
    c_dist = torch.matmul(query_states, key_centroids.transpose(2, 3))
    _, c_neighbor = torch.sort(c_dist, dim=-1, descending=True)
    # (num_heads, nlist)
    c_neighbor = c_neighbor.squeeze(0).squeeze(-2)
    neighbor_cluster_size = torch.gather(cluster_key_size, -1, c_neighbor)
    neighbor_cluster_key_size_ps = torch.cumsum(neighbor_cluster_size, dim=-1)
    # get the number of needed clusters by mask smaller and get min
    neighbor_cluster_key_size_ps[neighbor_cluster_key_size_ps < (token_budget-sink)] = 10000000
    # num_need_clusters: (num_kv_heads)
    _, num_need_clusters = torch.min(neighbor_cluster_key_size_ps, dim=-1)
    num_need_clusters += 1
    # now we select same number of clusters for all heads
    max_num_need_clusters = torch.max(num_need_clusters).item()

	# (num_heads, max_num_need_clusters)
    sel_cluster_indices = c_neighbor[:, :max_num_need_clusters]
    sel_cluster_size = neighbor_cluster_size[:, :max_num_need_clusters]
    # not use neighbor_cluster_key_size_ps[, :max_num_need_clusters] as it has be modified
    sel_cluster_size_ps = torch.cumsum(sel_cluster_size, dim=-1)
    sel_cluster_key_end = torch.gather(cluster_key_size_ps, -1, sel_cluster_indices)
    sel_cluster_key_start = sel_cluster_key_end - sel_cluster_size

    if cluster_cache is not None:
        cluster_params.update(sel_cluster_indices)
    
    # if self.layer_id == 10:
    #     print(neighbor_cluster_size[1])
    #     print(neighbor_cluster_key_size_ps[1])
    #     print(num_need_clusters[1])
    #     print(max_num_need_clusters)
    #     print(sel_cluster_indices[1])
    #     print(sel_cluster_size[1])
    #     print(sel_cluster_key_end[1])
    #     print(sel_cluster_key_start[1])
    #     print()
    use_search_kernel = True

    if use_search_kernel:
        max_num_indices = torch.sum(sel_cluster_size, dim=-1).max()
        sel_key_indices = torch.full((num_heads, max_num_indices), kv_seq_len, 
                                     dtype=torch.int64, device='cuda')
        search_indices(num_need_clusters,
                    sel_cluster_size_ps,
                    sel_cluster_key_start,
                    sel_cluster_key_end,
                    cluster_key_indices,
                    sel_key_indices)
        sel_key_indices = sel_key_indices[:, :token_budget-sink]
    else:
        sel_key_indices = []
        for h in range(num_heads):
            kv_h = h // num_key_value_groups
            head_num_need_clusters = num_need_clusters[h]
            head_sel_key_indices = []
            for i in range(head_num_need_clusters):
                head_sel_key_indices.append(cluster_key_indices[
                    kv_h, sel_cluster_key_start[h, i]: sel_cluster_key_end[h, i]
                ])
            head_sel_key_indices = torch.cat(head_sel_key_indices)
            # if self.layer_id == 10:
            #     print(head_sel_key_indices.shape)
            sel_key_indices.append(head_sel_key_indices)
        # sel_key_indices: (1, num_heads, token_budget) or 
        if head_sel == "pad":
            sel_key_indices = pad_sequence(sel_key_indices, batch_first=True, padding_value=kv_seq_len)
        elif head_sel == "truc":
            sel_key_indices = torch.stack([ind[:token_budget-sink] for ind in sel_key_indices])
        else:
            assert False
    
    sel_key_indices = sel_key_indices.unsqueeze(0)
    sel_key_indices += sink
    sel_key_indices[sel_key_indices > kv_seq_len] = kv_seq_len

    if topk_stat:
        sink_indices = torch.arange(sink, device=sel_key_indices.device).repeat(1, num_heads, 1)
        full_sel_key_indices = torch.cat([sink_indices, sel_key_indices], dim=-1)
        nlist = c_dist.shape[-1]
        assert cluster_params is not None
        stat_topk(layer_id, full_sel_key_indices, query_states, 
                  key_states[:, :, :prompt_len, :], 
                  f'nc{nlist}-fi{cluster_params.max_iter}')

    # sel_key_indices: (1, num_heads, token_budget, head_dim)
    sel_key_indices = sel_key_indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
    if head_sel == "truc":
        sel_key_states = key_states.gather(dim=2, index=sel_key_indices)
        sel_value_states = value_states.gather(dim=2, index=sel_key_indices)
    elif head_sel == "pad":
        kpad = torch.ones((key_states.shape[0], key_states.shape[1], 
                                1, key_states.shape[3]), dtype=key_states.dtype, 
                                device=key_states.device) * torch.finfo(key_states.dtype).min
        vpad = torch.zeros((value_states.shape[0], value_states.shape[1], 
                                1, value_states.shape[3]), dtype=value_states.dtype, 
                                device=value_states.device)
        sign = (query_states > 0) + (~(query_states > 0)) * -1
        sel_key_states = torch.cat([key_states, kpad*sign], dim=2).gather(dim=2, index=sel_key_indices)
        sel_value_states = torch.cat([value_states, vpad], dim=2).gather(dim=2, index=sel_key_indices)

    sel_key_states = torch.cat([key_states[:, :, :sink, :], sel_key_states, 
                                key_states[:, :, prompt_len:, :]], dim=2)
    sel_value_states = torch.cat([value_states[:, :, :sink, :], sel_value_states, 
                                  value_states[:, :, prompt_len:, :]], dim=2)

    # if self.layer_id == 10:
    #     print(sel_key_states.shape, sel_value_states.shape)
    attn_weights = torch.matmul(query_states, sel_key_states.transpose(2, 3)) / math.sqrt(head_dim)

    if attention_mask is not None:  # no matter the length, we just slice it
        causal_mask = attention_mask[:, :, :, : sel_key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    # attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
    attn_output = torch.matmul(attn_weights, sel_value_states)

    if attn_output.size() != (bsz, num_heads, q_len, head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, num_heads, q_len, head_dim)}, but is"
            f" {attn_output.size()}"
        )

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, hidden_size)
    return attn_output

def forward_cluster(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[DynamicCache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()
    assert bsz == 1

    if self.layer_id < 2 or q_len > 1 \
        or (self.prompt_len == 0 and q_len < self.token_budget) \
        or (self.prompt_len > 0 and self.prompt_len+q_len < self.token_budget) :   # for first several tokens of ppl_eval
        if q_len > 1:
            self.prompt_len = q_len
            # reset cache for each request
            if self.cache_steps > 0 and self.layer_id >= 2:
                self.cluster_cache = CacheSimulator(self.layer_id, self.cache_steps+1)
        return self.flash_forward(
            hidden_states,
            attention_mask,
            position_ids,
            past_key_value,
            output_attentions,
            use_cache,
            **kwargs,
        )
    
    sink = self.sink
    prefill_key = past_key_value[self.layer_id][0]
    assert prefill_key.shape[-2] > sink
    prefill_key = prefill_key[..., sink:, :]
    # clustering for prefilled keys
    if self.key_centroids is None:
        self.key_centroids, self.cluster_key_indices, \
        self.cluster_key_ptr, self.cluster_key_size, self.cluster_key_size_ps = \
		build_cluster(prefill_key, self.nlist, self.balance, self.cluster_params,
                    self.num_key_value_groups, self.gqa_policy)

    query_states = (
        self.q_proj(hidden_states)
        .view(bsz, q_len, self.num_heads, self.head_dim)
        .transpose(1, 2)
    )
    key_states = (
        self.k_proj(hidden_states)
        .view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        .transpose(1, 2)
    )
    value_states = (
        self.v_proj(hidden_states)
        .view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        .transpose(1, 2)
    )

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value[self.layer_id][0].shape[-2]
    cos, sin = self.rotary_emb(value_states, position_ids)
    query_states, key_states = apply_rotary_pos_emb(
        query_states, key_states, cos, sin, position_ids
    )
    # [bsz, nh, t, hd]

    if past_key_value is not None:
        # reuse k, v, self_attention
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_id)

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    token_budget = min(self.prompt_len, self.token_budget)
    attn_output = cluster_attn_out(
        query_states, key_states, value_states, attention_mask, 
        self.prompt_len, self.key_centroids, self.cluster_key_indices, 
        self.cluster_key_size, self.cluster_key_size_ps,
        self.num_key_value_groups, self.layer_id, token_budget, 
        sink, self.head_sel, self.cluster_cache, self.topk_stat, self.cluster_params
    )
    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value

def split_tensor_along_last_dim(
        tensor: torch.Tensor,
        num_partitions: int,
        contiguous_split_chunks: bool = False,
) -> List[torch.Tensor]:
    """Split a tensor along its last dimension.

    Arguments:
        tensor: input tensor.
        num_partitions: number of partitions to split the tensor
        contiguous_split_chunks: If True, make each chunk contiguous
                                 in memory.

    Returns:
        A list of Tensors
    """
    # Get the size and dimension.
    last_dim = tensor.dim() - 1
    last_dim_size = tensor.size()[last_dim] // num_partitions
    # Split.
    tensor_list = torch.split(tensor, last_dim_size, dim=last_dim)
    # Note: torch.split does not create contiguous tensors by default.
    if contiguous_split_chunks:
        return tuple(chunk.contiguous() for chunk in tensor_list)

    return tensor_list

@torch.jit.script
def glm_apply_rotary_pos_emb(x: torch.Tensor, rope_cache: torch.Tensor) -> torch.Tensor:
    # x: [b, np, sq, hn]
    b, np, sq, hn = x.size(0), x.size(1), x.size(2), x.size(3)
    rot_dim = rope_cache.shape[-2] * 2
    x, x_pass = x[..., :rot_dim], x[..., rot_dim:]
    # truncate to support variable sizes
    rope_cache = rope_cache[:, :sq]
    xshaped = x.reshape(b, np, sq, rot_dim // 2, 2)
    rope_cache = rope_cache.view(-1, 1, sq, xshaped.size(3), 2)
    x_out2 = torch.stack(
        [
            xshaped[..., 0] * rope_cache[..., 0] - xshaped[..., 1] * rope_cache[..., 1],
            xshaped[..., 1] * rope_cache[..., 0] + xshaped[..., 0] * rope_cache[..., 1],
        ],
        -1,
    )
    x_out2 = x_out2.flatten(3)
    return torch.cat((x_out2, x_pass), dim=-1)

def forward_cluster_glm(
    self, hidden_states, attention_mask, rotary_pos_emb, kv_cache=None, use_cache=True
):
    bsz, q_len, _ = hidden_states.size()
    assert bsz == 1

    if q_len > 1 or self.layer_number < 3 \
        or (self.prompt_len == 0 and q_len < self.token_budget) \
        or (self.prompt_len > 0 and self.prompt_len+q_len < self.token_budget) :   # for first several tokens of ppl_eval
        if q_len > 1:
            self.prompt_len = q_len
            if self.cache_steps > 0 and self.layer_number>= 2:
                self.cluster_cache = CacheSimulator(self.layer_number, self.cache_steps+1)
        return self.flash_forward(
            hidden_states,
            attention_mask,
            rotary_pos_emb,
            kv_cache,
            use_cache
        )

    sink = self.sink
    prefill_key = kv_cache[0]
    prefill_key = prefill_key[..., sink:, :]
    num_key_value_group = self.num_attention_heads_per_partition // self.num_multi_query_groups_per_partition
    # clustering for prefilled keys
    if self.key_centroids is None:
        self.key_centroids, self.cluster_key_indices, \
        self.cluster_key_ptr, self.cluster_key_size, self.cluster_key_size_ps = \
		build_cluster(prefill_key, self.nlist, self.balance, self.cluster_params, 
                    num_key_value_group, self.gqa_policy)
    # hidden_states: [b, sq, h]

    # =================================================
    # Pre-allocate memory for key-values for inference.
    # =================================================
    # =====================
    # Query, Key, and Value
    # =====================

    # Attention heads [b, sq, h] --> [b, sq, (np * 3 * hn)]
    mixed_x_layer = self.query_key_value(hidden_states)

    if self.multi_query_attention:
        (query_layer, key_layer, value_layer) = mixed_x_layer.split(
            [
                self.num_attention_heads_per_partition * self.hidden_size_per_attention_head,
                self.num_multi_query_groups_per_partition * self.hidden_size_per_attention_head,
                self.num_multi_query_groups_per_partition * self.hidden_size_per_attention_head,
            ],
            dim=-1,
        )
        query_layer = query_layer.view(
            query_layer.size()[:-1] + (self.num_attention_heads_per_partition, self.hidden_size_per_attention_head)
        )
        key_layer = key_layer.view(
            key_layer.size()[:-1] + (self.num_multi_query_groups_per_partition, self.hidden_size_per_attention_head)
        )
        value_layer = value_layer.view(
            value_layer.size()[:-1]
            + (self.num_multi_query_groups_per_partition, self.hidden_size_per_attention_head)
        )
    else:
        new_tensor_shape = mixed_x_layer.size()[:-1] + \
                            (self.num_attention_heads_per_partition,
                            3 * self.hidden_size_per_attention_head)
        mixed_x_layer = mixed_x_layer.view(*new_tensor_shape)

        # [b, sq, np, 3 * hn] --> 3 [b, sq, np, hn]
        (query_layer, key_layer, value_layer) = split_tensor_along_last_dim(mixed_x_layer, 3)

    # [b, sq, np, hn] -> [b, np, sq, hn]
    query_layer, key_layer, value_layer = [k.transpose(1, 2) for k in [query_layer, key_layer, value_layer]]

    # apply relative positional encoding (rotary embedding)
    if rotary_pos_emb is not None:
        query_layer = glm_apply_rotary_pos_emb(query_layer, rotary_pos_emb)
        key_layer = glm_apply_rotary_pos_emb(key_layer, rotary_pos_emb)

    # adjust key and value for inference
    if kv_cache is not None:
        cache_k, cache_v = kv_cache
        key_layer = torch.cat((cache_k, key_layer), dim=2)
        value_layer = torch.cat((cache_v, value_layer), dim=2)
    if use_cache:
        if kv_cache is None:
            kv_cache = torch.cat((key_layer.unsqueeze(0).unsqueeze(0), value_layer.unsqueeze(0).unsqueeze(0)),
                                    dim=1)
        else:
            kv_cache = (key_layer, value_layer)
    else:
        kv_cache = None

    if self.multi_query_attention:
        key_layer = key_layer.unsqueeze(2)
        key_layer = key_layer.expand(
            -1, -1, self.num_attention_heads_per_partition // self.num_multi_query_groups_per_partition, -1, -1
        )
        key_layer = key_layer.contiguous().view(
            key_layer.size()[:1] + (self.num_attention_heads_per_partition,) + key_layer.size()[3:]
        )
        value_layer = value_layer.unsqueeze(2)
        value_layer = value_layer.expand(
            -1, -1, self.num_attention_heads_per_partition // self.num_multi_query_groups_per_partition, -1, -1
        )
        value_layer = value_layer.contiguous().view(
            value_layer.size()[:1] + (self.num_attention_heads_per_partition,) + value_layer.size()[3:]
        )

    # ==================================
    # core attention computation
    # ==================================

    # context_layer = self.core_attention(query_layer, key_layer, value_layer, attention_mask)
    token_budget = min(self.prompt_len, self.token_budget)
    context_layer = cluster_attn_out(
        query_layer, key_layer, value_layer, attention_mask, 
        self.prompt_len, self.key_centroids, self.cluster_key_indices, 
        self.cluster_key_size, self.cluster_key_size_ps,
        num_key_value_group, self.layer_number, token_budget, 
        sink, self.head_sel, self.cluster_cache, self.topk_stat, self.cluster_params
    )

    # =================
    # Output. [sq, b, h]
    # =================

    output = self.dense(context_layer)

    return output, kv_cache

MAX_POOL_SIZE = 1*1024**3
def cluster_reset(model):
    if isinstance(model, PreTrainedModel):
        rmm.reinitialize(pool_allocator=True, initial_pool_size=MAX_POOL_SIZE, maximum_pool_size=MAX_POOL_SIZE)
        torch.cuda.empty_cache()
    for name, module in reversed(model._modules.items()):
        if len(list(module.children())) > 0:
            cluster_reset(module)
        module.key_centroids = None
        module.cluster_key_indices = None
        module.cluster_key_ptr = None
        module.cluster_key_size = None
        module.cluster_key_size_ps = None

def apply_cluster_config(module, args):
    nlist = args.nlist
    module.nlist = nlist
    module.head_sel = args.head_sel
    module.balance = True if args.balance else False
    module.sink = args.sink
    module.gqa_policy = args.gqa_policy
    if args.balance:
        module.cluster_params = ivf_flat.IndexParams(
        n_lists=nlist, metric='inner_product', kmeans_n_iters=args.fit_iter,
        kmeans_trainset_fraction=1, add_data_on_build=False)
    else:
        module.cluster_params = KMeansParams(
            n_clusters=nlist, max_iter=args.fit_iter, metric=args.dist_t)
