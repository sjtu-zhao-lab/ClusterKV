import torch
from typing import Optional
import flashinfer

import clusterkv._clusterkv_knl as _kernels
from clusterkv.clusterkv_utils.clusterkv_controller import ClusterKVController
from clusterkv.clusterkv_utils.decode_wrapper import BatchDecodeWithPagedKVCacheWrapper

__all__ = [
    'ClusterKVController',
    'append_kv',
    "BatchDecodeWithPagedKVCacheWrapper",
    "prefill_forward",
    "build_cluster",
    "decode_sparse_attn",
    "update_sel_indices"
]


def append_kv(
    k: torch.Tensor, v: torch.Tensor,
    controller: ClusterKVController, layer_idx: int,
):
    """
    Semantics of `append_kv`:
    Append new generated k/v into kv cache and meta data cache.
    Automatically dispatch to Prefill / Decode Kernel

    Notations for shapes:
    `B`: batch size
    `N`: number of heads
    `D`: head dimension
    `L`: number of layers
    `MAXLEN`: maximum length of the KV cache

    Args:
        k: Shape: `[L, N, D]`. Key projection (`X @ W_k`).
        v: Shape: `[L, N, D]`. Value projection (`X @ W_v`).
        controller: InferenceController object, which contains all needed information.
        layer_idx: Layer index of the KV cache.
    """
    seq_len = k.size(0)
    if seq_len > 1:
        # if layer_idx < 3:
        #     print("Prefill append", layer_idx)
        #     print(controller.kv_indices_with_last, controller.kv_indptr_for_append,
        #             controller.kv_last_page_idx)
        #     print(controller.kv_indices_with_last_offload, controller.kv_indptr_for_append_offload,
        #             controller.kv_last_page_idx_offload)
        if controller.offload and layer_idx >= 2:
            token_budget = controller._token_budget
            sink = controller.sink
            _kernels.append_kv_cache_prefill(
                k[:sink + token_budget, ...].repeat_interleave(controller.num_key_value_groups, dim=1),
                v[:sink + token_budget, ...].repeat_interleave(controller.num_key_value_groups, dim=1),
                controller.kv_cache[layer_idx],
                controller.kv_indices_with_last_offload,
                controller.kv_indptr_for_append_offload,
                controller.last_page_len,
                controller.kv_last_page_idx_offload,
                controller.layout
            )
        else:
            _kernels.append_kv_cache_prefill(
                k,
                v,
                controller.kv_cache[layer_idx],
                controller.kv_indices_with_last,
                controller.kv_indptr_for_append,
                controller.last_page_len,
                controller.kv_last_page_idx,
                controller.layout
            )
    else:
        # if layer_idx < 3:
        #     print("Decode append", layer_idx)
        #     print(controller.kv_indices_with_last_offload, controller.kv_indptr_for_append_offload,
        #           controller.kv_last_page_idx_offload)
        if controller.offload and layer_idx >= 2:
            token_budget = controller._token_budget
            _kernels.append_kv_cache_decode(
                k.repeat_interleave(controller.num_key_value_groups, dim=1),
                v.repeat_interleave(controller.num_key_value_groups, dim=1),
                controller.kv_cache[layer_idx],
                controller.kv_indices_with_last_offload,
                controller.kv_indptr_for_append_offload,
                controller.last_page_len,
                controller.kv_last_page_idx_offload,
                controller.layout
            )
        else:
            _kernels.append_kv_cache_decode(
                k,
                v,
                controller.kv_cache[layer_idx],
                controller.kv_indices_with_last,
                controller.kv_indptr_for_append,
                controller.last_page_len,
                controller.kv_last_page_idx,
                controller.layout
            )


def prefill_forward(
    q: torch.Tensor,
    controller: ClusterKVController,
    layer_idx: int,
    rope_scale: Optional[float] = None,
    rope_theta: Optional[float] = None,
    key_states: Optional[torch.Tensor] = None,
    value_states: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if rope_scale is None:
        rope_scale = 1.0
    if rope_theta is None:
        rope_theta = 1e4
    # key_states, value_states = controller.get_kv(layer_idx)
    # key_states = key_states.squeeze(1).contiguous()   # [prompt_len, num_kv_heads, head_dim]
    # value_states = value_states.squeeze(1).contiguous()   # [prompt_len, num_kv_heads, head_dim]
    if key_states is not None:
        o = flashinfer.single_prefill_with_kv_cache(q, key_states, value_states, 
                                                    causal=True, use_fp16_qk_reduction=False,
                                                    rope_scale=rope_scale, rope_theta=rope_theta)
    else:
        f = _kernels.prefill_with_paged_kv_cache
        o = f(
            q,
            controller.kv_cache[layer_idx],
            controller.kv_indices_with_last,
            controller.last_page_len,
            True, # Casual
            controller.layout,
            False, # FP16 Accumulator for 4090
            rope_scale,
            rope_theta,
        )
    return o


def build_cluster(controller: ClusterKVController, 
                  layer_idx: int, 
                  keys: torch.Tensor,
                  key_offset: int,
                  nlist: int,
                  stream: torch.cuda.Stream):
    keys = keys.transpose(0, 1).contiguous()
    num_kv_heads, seq_len, head_dim = keys.shape
    niter = controller.niter
    init_idx = torch.randint(0, seq_len, (nlist,))
    centroids = keys[:, init_idx, :]                        # [num_kv_heads, nlist, head_dim]
    labels = controller.get_labels(seq_len)                 # [num_kv_heads, seq_len]
    cluster_size = torch.zeros((num_kv_heads, nlist), dtype=torch.int32, device=keys.device)
    epsilon = 1e-8
    keys_normalized = keys / (torch.linalg.norm(keys, dim=-1, keepdim=True) + epsilon)
    for _ in range(niter):
        centroids_normalized = centroids / (torch.linalg.norm(centroids, dim=-1, keepdim=True) + epsilon)
        # [num_kv_heads, seq_len, n_list]
        cos_sim = torch.matmul(keys_normalized, centroids_normalized.transpose(1, 2))
        labels[:, :seq_len] = cos_sim.argmax(dim=2)
        _kernels.update_centroids(keys, labels, centroids, 
                                     controller.max_seq_len, stream.cuda_stream)
    centroids_normalized = centroids / (torch.linalg.norm(centroids, dim=-1, keepdim=True) + epsilon)
    cos_sim = torch.matmul(keys_normalized, centroids_normalized.transpose(1, 2))
    labels[:, :seq_len] = cos_sim.argmax(dim=2)
    _kernels.count_labels(labels, cluster_size, controller.max_seq_len, stream.cuda_stream)
    cluster_size_ps = torch.cumsum(cluster_size, dim=-1)
    _, ind = torch.sort(labels)
    if key_offset > 0:
        ind += key_offset
    # update metadata in controller
    controller.update_metadata(layer_idx, centroids, cluster_size, cluster_size_ps, ind)


def check_uniqueness(tensor: torch.Tensor, name: str):
    # tensor is 2D: [num_heads, budget]
    for i, row in enumerate(tensor):
        unique_elements = torch.unique(row)
        if len(unique_elements) != len(row):
            raise ValueError(f"Tensor '{name}' has duplicate elements in row {i}: {sorted(row.tolist())}")


def update_sel_indices(
    q: torch.Tensor,            # [q_len=1, num_heads, head_dim]
    controller: ClusterKVController,
    layer_idx: int,
):
    _kernels.get_neigh_c(q, controller.centroids[layer_idx], 
                            controller.cluster_size[layer_idx], 
                            controller.neigh_c, controller.neigh_c_size)

    _kernels.get_sel_indices(controller.neigh_c, controller.neigh_c_size, 
                                controller.cluster_size_ps[layer_idx],
                                controller.cluster_key_indices[layer_idx], 
                                controller.sel_token_indices)

    controller.sel_token_indices += controller.sink
    

def decode_sparse_attn(
    q: torch.Tensor,
    controller: ClusterKVController,
    layer_idx: int,
    topk_indices: torch.Tensor,
    rope_scale: Optional[float] = None,
    rope_theta: Optional[float] = None,
) -> torch.Tensor:
    """
    Semantics of `decode_sparse_attn`:
    Excute self-attention only on the selected pages (Top-k output)

    Notations for shapes:
    `B`: batch size
    `N`: number of heads
    `D`: head dimension
    `L`: number of layers
    `MAXLEN`: maximum length of the KV cache

    Args:
        q: Shape: `[B, N, D]`. Key projection (`X @ W_k`).
        iController: InferenceController object, which contains all needed information.
        layer_idx: Layer index of the KV cache.
        topk_indices: Shape: `[N, page_budget-1]`. Top-k indices.
    """
    o = torch.empty_like(q, dtype=q.dtype, device=q.device)
    # if layer_idx < 3:
    #     print(layer_idx, controller.kv_indptr_for_approx_decode, controller.kv_last_page_idx)
    #     print(layer_idx, controller.kv_indptr_for_approx_decode, controller.kv_last_page_idx_offload)
    # if topk_indices is not None:
    #     assert torch.all(topk_indices >= 0)
    #     assert torch.all(topk_indices < controller.kv_seqlen)

    if controller.offload and layer_idx >= 2:
        # recall_impl = "naive"
        recall_impl = ""
        if recall_impl == "naive":
            num_heads, budget = topk_indices.shape 
            # [K, 1, num_heads, 1]
            topk_indices = topk_indices.transpose(0, 1).unsqueeze(1).unsqueeze(3)
            # [K, 2, num_heads, head_dim]
            topk_indices = topk_indices.expand(budget, 2, num_heads, controller.head_dim)
            controller.default_stream.wait_event(controller.offload_events[layer_idx])
            cpu_select_kv = torch.gather(
                controller.kv_cache_cpu[layer_idx].repeat_interleave(controller.num_key_value_groups, dim=-2), 
                0, topk_indices.to("cpu").to(torch.int64))
            kv_cache_mid = controller.kv_cache_mid(layer_idx)
            kv_cache_mid.copy_(cpu_select_kv)
        else:
            _kernels.recall(
                controller.kv_cache_mid(layer_idx),
                controller.kv_cache_cpu[layer_idx],
                topk_indices,
                controller.g2c[layer_idx],
                controller.c2g,
                controller.is_in_cache,
                controller.is_in_topk,
                controller.swap_out_indices,
                controller.swap_in_indices,
                controller.swap_out_count,
                controller.swap_in_count,
                controller.kv_seqlen,
            )
            # assert(torch.all(controller.swap_out_count == controller.swap_in_count))

        controller._decode_handler.forward(
            q,
            o,
            controller.kv_cache[layer_idx],
            controller.empty_data,
            # torch.arange(0, controller.kv_last_page_idx_offload+1, device="cuda", dtype=torch.int32).repeat(controller.num_heads, 1),
            controller.kv_indptr_for_approx_decode,
            controller.last_page_len,
            controller.kv_last_page_idx_offload,
            rope_scale,
            rope_theta,
        )
    else:
        if topk_indices is not None:
            topk_indices = torch.cat([
                controller.sink_indices, topk_indices, controller.cur_win_indices
            ], dim=-1)

        controller._decode_handler.forward(
            q,
            o,
            controller.kv_cache[layer_idx],
            topk_indices if topk_indices is not None else controller.empty_data,
            controller.kv_indptr_for_approx_decode,
            controller.last_page_len,
            controller.kv_last_page_idx,
            rope_scale,
            rope_theta,
        )
    return o
