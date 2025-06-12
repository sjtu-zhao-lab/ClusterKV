import math
from typing import Optional, Tuple, List

import torch
from torch import nn

from transformers.models.llama.modeling_llama import (
    apply_rotary_pos_emb,
    repeat_kv,
)
from transformers.cache_utils import DynamicCache

from .cluster_cache_simulator import CacheSimulator

def local_heavy_hitter_mask(attn_weights, token_budget, chunk_size):
    # attn_weights (BS, head, query, keys)

    # expend attn_weights to be divisible by chunk_size
    seq_length = attn_weights.shape[-1]
    padding_length = chunk_size - ((seq_length - 1) % chunk_size + 1)
    attn_weights = torch.cat(
        [
            attn_weights,
            torch.ones(
                (
                    attn_weights.shape[0],
                    attn_weights.shape[1],
                    attn_weights.shape[2],
                    padding_length,
                ),
                device=attn_weights.device,
            )
            * torch.tensor(torch.finfo(attn_weights.dtype).min),
        ],
        dim=-1,
    )

    # chunk attn_weights into chunk_size tokens
    chunk_attn_weights = attn_weights.reshape(
        attn_weights.shape[0],
        attn_weights.shape[1],
        attn_weights.shape[2],
        attn_weights.shape[3] // chunk_size,
        chunk_size,
    ).amax(dim=-1)

    _, topk_page = chunk_attn_weights.topk(
        k=min(max(3, token_budget // chunk_size), chunk_attn_weights.size(-1)), dim=-1
    )
    # repeat topk chunk_size times and recover the original indexes (* chunk_size + arange(chunk_size))
    topk = topk_page.unsqueeze(-1).repeat(
        1, 1, 1, 1, chunk_size
    ) * chunk_size + torch.arange(chunk_size, device=topk_page.device)
    topk = topk.reshape(topk.shape[0], topk.shape[1], topk.shape[2], -1)
    mask_bottom = torch.zeros_like(attn_weights, dtype=torch.bool)
    mask_bottom.scatter_(-1, topk, True)

    # remove the padding
    mask_bottom_full = mask_bottom
    mask_bottom = mask_bottom[:, :, :, :seq_length]

    return mask_bottom, topk_page, mask_bottom_full

def stat_topk(layer_id, mask: torch.Tensor, attn_weights: torch.Tensor, name):
    _, num_heads, _, prompt_len = mask.shape
    mask_nz_indices = torch.nonzero(mask)
    assert mask_nz_indices.shape[0] % num_heads == 0
    k = mask_nz_indices.shape[0] // num_heads
    indices = torch.zeros(1, num_heads, 1, k, device=mask_nz_indices.device, 
                          dtype=mask_nz_indices.dtype)
    head_indices, seq_len_indices = mask_nz_indices[:, 1], mask_nz_indices[:, 3]
    seq_arange = torch.arange(k, device=mask.device).repeat(num_heads)
    indices[0, head_indices, 0, seq_arange] = seq_len_indices
    _, topk_indices = attn_weights[..., :prompt_len].topk(k, dim=-1)
    hit_rate = []
    for h in range(num_heads):
        truth = topk_indices[0, h, 0].cpu()
        pred = indices[0, h, 0].cpu()
        hit_rate.append(len( set(truth.numpy()) & set(pred.numpy()) ) / k)
    avg_hit_rate = sum(hit_rate) / len(hit_rate)
    with open(f'topk_stat/top{k}-{name}.log', 'a') as f:
        f.write(f'layer {layer_id}: {avg_hit_rate}\n')

def quest_attn_out(query_states, key_states, value_states, attention_mask, 
                   keep_gen, prompt_len, chunk_size, token_budget, layer_id,
                   cluster_cache, topk_stat=False):
    bsz, num_heads, q_len, head_dim = query_states.shape
    hidden_size = num_heads * head_dim
    kv_seq_len = key_states.shape[-2]
    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(head_dim)

    sign = (query_states > 0) + (~(query_states > 0)) * -1
    max_key = key_states * sign
    if keep_gen:
        max_key = max_key[:, :, :prompt_len, :]
    postive_query = query_states * sign

    # expend max_key to be divisible by chunk_size
    seq_length = max_key.shape[-2]
    padding_length = chunk_size - ((seq_length - 1) % chunk_size + 1)
    max_key = torch.cat(
        [
            max_key,
            torch.ones(
                (max_key.shape[0], max_key.shape[1], padding_length, max_key.shape[3]),
                device=max_key.device,
            )
            * torch.tensor(torch.finfo(max_key.dtype).min),
        ],
        dim=-2,
    )

    # chunk max_key into chunk_size tokens
    chunk_max_key = max_key.reshape(
        max_key.shape[0],
        max_key.shape[1],
        max_key.shape[2] // chunk_size,
        chunk_size,
        max_key.shape[3],
    ).amax(dim=-2)

    # duplicate chunk_max_key chunk_size times
    chunk_max_key = chunk_max_key.unsqueeze(-2).repeat(1, 1, 1, chunk_size, 1)
    # reshape chunk_max_key to the original shape
    chunk_max_key = chunk_max_key.reshape(
        chunk_max_key.shape[0], chunk_max_key.shape[1], -1, chunk_max_key.shape[-1]
    )[:, :, :seq_length, :]

    quantized_weight = torch.matmul(
        postive_query.float(),
        chunk_max_key.transpose(2, 3),
    )

    if attn_weights.size() != (bsz, num_heads, q_len, kv_seq_len):
        raise ValueError(
            f"Attention weights should be of size {(bsz, num_heads, q_len, kv_seq_len)}, but is"
            f" {attn_weights.size()}"
        )

    if attention_mask is not None:
        if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
            raise ValueError(
                f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
            )
        attn_weights = attn_weights + attention_mask
        attn_weights = torch.max(
            attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min)
        )
        # when ksg, the attn_mask is longer than quantized_weight
        quantized_weight = quantized_weight + attention_mask[..., :quantized_weight.shape[-1]]
        quantized_weight = torch.max(
            quantized_weight, torch.tensor(torch.finfo(quantized_weight.dtype).min)
        )

    token_budget = min(prompt_len, token_budget) if keep_gen else min(kv_seq_len, token_budget)

    attn_weights_for_selection = quantized_weight

    if token_budget > 0:
        mask_bottom, topk_page, mask_bottom_full = local_heavy_hitter_mask(
            attn_weights_for_selection, token_budget, chunk_size
        )  # Default: No padding applied to input
        if cluster_cache is not None:
            # [1, num_heads, 1, num_pages] to [num_heads, num_pages]
            topk_page = topk_page.squeeze(0).squeeze(1)
            cluster_cache.update(topk_page)
    else:
        mask_bottom = torch.zeros_like(attn_weights_for_selection, dtype=torch.bool)
    
    if topk_stat:
        assert keep_gen
        # print(layer_id, mask_bottom.sum(dim=(0,2,3)), mask_bottom_full.sum(dim=(0,2,3)),)
        stat_topk(layer_id, mask_bottom_full, attn_weights, f"qg{chunk_size}")
    
    if keep_gen:
        gen_true_mask = torch.ones((mask_bottom.shape[0], mask_bottom.shape[1],
                                    mask_bottom.shape[2], kv_seq_len - mask_bottom.shape[3]),
                                    dtype=torch.bool, device=mask_bottom.device)
        mask_bottom = torch.cat([mask_bottom, gen_true_mask], dim=-1)
    mask_bottom = torch.tril(mask_bottom, diagonal=kv_seq_len-1)
    # if self.layer_id == 10:
    #     print(mask_bottom.shape, mask_bottom.sum(dim=-1))
        # print(mask_bottom[..., self.prompt_len:])
    attn_weights[~mask_bottom] = torch.tensor(torch.finfo(attn_weights.dtype).min)

    # upcast attention to fp32
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
        query_states.dtype
    )
    attn_output = torch.matmul(attn_weights, value_states)

    if attn_output.size() != (bsz, num_heads, q_len, head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, num_heads, q_len, head_dim)}, but is"
            f" {attn_output.size()}"
        )

    attn_output = attn_output.transpose(1, 2)
    attn_output = attn_output.reshape(bsz, q_len, hidden_size)
    return attn_output
    

def forward_quest(
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

    attn_output = quest_attn_out(query_states, key_states, value_states, attention_mask,
                                 self.gen, self.prompt_len, self.chunk_size,
                                 self.token_budget, self.layer_id, 
                                 self.cluster_cache, self.topk_stat)
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

def forward_quest_glm(
    self, hidden_states, attention_mask, rotary_pos_emb, kv_cache=None, use_cache=True
):
    _, q_len, _ = hidden_states.size()
    if self.layer_number < 3 or q_len > 1 \
        or (self.prompt_len == 0 and q_len < self.token_budget) \
        or (self.prompt_len > 0 and self.prompt_len+q_len < self.token_budget) :   # for first several tokens of ppl_eval
        if q_len > 1:
            self.prompt_len = q_len
            if self.cache_steps > 0 and self.layer_number >= 3:
                self.cluster_cache = CacheSimulator(self.layer_number, self.cache_steps+1)
        return self.flash_forward(
            hidden_states,
            attention_mask,
            rotary_pos_emb,
            kv_cache,
            use_cache
        )
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
    context_layer = quest_attn_out(query_layer, key_layer, value_layer, attention_mask,
                                   self.gen, self.prompt_len, self.chunk_size, 
                                   self.token_budget, self.layer_number, 
                                   self.cluster_cache, self.topk_stat)

    # =================
    # Output. [sq, b, h]
    # =================

    output = self.dense(context_layer)

    return output, kv_cache
