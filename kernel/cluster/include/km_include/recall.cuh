#include <cuda.h>

__global__ void swap_in_out_kernel(
    int* g2c,
    const int* topk,
    int* c2g_buf,
    bool* is_in_cache_buf,
    bool* is_in_topk_buf,
    int* swap_out_indices,
    int* swap_in_indices,
    int* swap_out_count,
    int* swap_in_count,
    int budget,
    int seq_len,
    int max_seqlen
) {
    const int blcok_size = blockDim.x;
    const int head = blockIdx.x;
    const int tid = threadIdx.x;
    
    extern __shared__ int s_mem[];
    int* s_swap_out_indices = s_mem;
    int* s_swap_in_indices = s_swap_out_indices + budget;
    int* s_swap_out_count = s_swap_in_indices + budget; // A single int
    int* s_swap_in_count = s_swap_out_count + 1;       // A single int

    bool* is_in_cache = is_in_cache_buf + head * max_seqlen;
    bool* is_in_topk = is_in_topk_buf + head * max_seqlen;
    int* c2g = c2g_buf + head * max_seqlen;

    if (tid == 0) {
        *s_swap_out_count = 0;
        *s_swap_in_count = 0;
    }
    for (int i = 0; i*blcok_size < seq_len; i++) {
        // init swaps
        int idx = i*blcok_size + tid;
        if (idx < seq_len) {
            is_in_cache[idx] = 0;
            is_in_topk[idx] = 0;
        }
    }
    __syncthreads();

    for (int i = 0; i*blcok_size < budget; i++) {
        int idx = i*blcok_size + tid;
        if (idx < budget) {
            const int token_idx = g2c[head*budget + idx];
            if (token_idx < seq_len) {
                is_in_cache[token_idx] = 1;
                c2g[token_idx] = idx;
            }
            const int topk_token_idx = topk[head*budget + idx];
            if (topk_token_idx < seq_len) {
                is_in_topk[topk_token_idx] = 1;
            }
        }
    }
    __syncthreads();
    
    for (int i = 0; i*blcok_size < seq_len; i++) {
        // init swaps
        int idx = i*blcok_size + tid;
        if (idx < seq_len) {
            bool should_swap_out = is_in_cache[idx] && !is_in_topk[idx];
            bool should_swap_in = is_in_topk[idx] && !is_in_cache[idx];
            if (should_swap_out) {
                int pos = atomicAdd(s_swap_out_count, 1);
                // Ensure we don't write out of bounds in shared memory
                if (pos < budget) {
                    // s_swap_out_indices[pos] = idx;
                    // use the index of cache for swap out
                    s_swap_out_indices[pos] = c2g[idx];
                }
            }
            if (should_swap_in) {
                int pos = atomicAdd(s_swap_in_count, 1);
                // Ensure we don't write out of bounds in shared memory
                if (pos < budget) {
                    s_swap_in_indices[pos] = idx;
                }
            }
        }
    }
    __syncthreads();

    // Write the compacted indices from shared to global memory
    if (tid == 0) {
        // assert(*s_swap_out_count == *s_swap_in_count);
        swap_out_count[head] = *s_swap_out_count;
        swap_in_count[head] = *s_swap_in_count;
    }

    for (int i = tid; i < *s_swap_out_count; i += blcok_size) {
        if (i < budget) { // Safety check
            int cache_slot_to_update = s_swap_out_indices[i];
            int new_token_to_load = s_swap_in_indices[i];

            // 1. Update the g2c map: the freed slot now points to the new token.
            g2c[head*budget + cache_slot_to_update] = new_token_to_load;

            // 2. Write out the indices for the host/data-copy kernel to use.
            swap_out_indices[head*budget + i] = cache_slot_to_update;
            swap_in_indices[head*budget + i] = new_token_to_load;
        }
    }
}


template <typename scalar_t>
__global__ void copy_kernel(
    scalar_t *__restrict__ gpu_buffer,
    const scalar_t *__restrict__ cpu_buffer, 
    const int* swap_out_indices_buf,
    const int* swap_in_indices_buf,
    const int* swap_count_buf,
    int group_size,
    int budget,
    int gpu_seq_stride,
    int gpu_kv_stride,
    int cpu_seq_stride,
    int cpu_kv_stride
) {
    int head = blockIdx.x;
    int kv_head = head / group_size;
    int seq_size = blockDim.x;
    int head_dim = blockDim.y;
    int seq_id = threadIdx.x;
    int dim_id = threadIdx.y;

    const int* swap_out_indices = swap_out_indices_buf + head*budget;
    const int* swap_in_indices = swap_in_indices_buf + head*budget;
    const int swap_count = swap_count_buf[head];
    for (int s = 0; s < swap_count; s += seq_size) {
        if (s + seq_id < swap_count) {
            int swap_out_i = swap_out_indices[s + seq_id];
            int swap_in_i = swap_in_indices[s + seq_id];
            gpu_buffer[swap_out_i*gpu_seq_stride + 0 + head*head_dim + dim_id]
                = cpu_buffer[swap_in_i*cpu_seq_stride + 0 + kv_head*head_dim + dim_id];
            gpu_buffer[swap_out_i*gpu_seq_stride + gpu_kv_stride + head*head_dim + dim_id]
                = cpu_buffer[swap_in_i*cpu_seq_stride + cpu_kv_stride + kv_head*head_dim + dim_id];
        }
    }
}