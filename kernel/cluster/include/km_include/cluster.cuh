#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h> // Include this for half precision support
#include <cub/cub.cuh>

template <typename scalar_t, int channel_split=2, bool count_only>
__global__ void update_centroids_kernel(
    const scalar_t* keys,
    const int* labels,
    scalar_t* centroids,
    int* label_counts,
    int num_kv_heads,
    int seq_len,
    int head_dim,
    int nlist,
    int max_seq_len
) {
	// return;
	const int head_stride = head_dim / channel_split;
    int kv_head = blockIdx.x;  // Each block corresponds to a kv_head
    if (kv_head >= num_kv_heads) return;

    scalar_t* sum;
    int* count;
    // Initialize shared memory for counts and sums
    if constexpr (!count_only) {
        // Shared memory for counting and accumulating values
        extern __shared__ scalar_t shared_memory[];
        sum = shared_memory;  // Sum of keys for each label
        count = (int*)&sum[nlist * head_stride]; // Count of occurrences for each label
    }
    else {
        extern __shared__ int shared_memory_cnt[];
        count = shared_memory_cnt;
    }

	int tid = threadIdx.x * blockDim.y + threadIdx.y;
	int thread_count = blockDim.x * blockDim.y;
    // Initialize sums and counts to zero
    if constexpr (!count_only) {
        for (int i = tid; i < nlist * head_stride; i += thread_count) {
            if (i < nlist * head_stride) {
                sum[i] = static_cast<scalar_t>(0);
            }
        }
    }
    for (int i = tid; i < nlist; i += thread_count) {
        count[i] = 0;
    }
    __syncthreads();

    // Loop over each key in this kv_head
    for (int i = threadIdx.y; i < seq_len; i += blockDim.y) {
        int label = labels[kv_head * max_seq_len + i];
        if (label >= 0 && label < nlist) {
            if constexpr (!count_only) {
                int j = threadIdx.x;
                atomicAdd(&sum[label * head_stride + j], 
                    keys[(kv_head * seq_len + i) * head_dim + blockIdx.y * head_stride + j]);
            }
            atomicAdd(&count[label], 1);
        }
        else {
            assert(false);
        }
    }
    __syncthreads();

    // Update centroids
    for (int i = threadIdx.y; i < nlist; i += blockDim.y) {
        if constexpr (!count_only) {
            if (count[i] > 0) {
                int j = threadIdx.x;
                centroids[(kv_head * nlist + i) * head_dim + blockIdx.y * head_stride + j] 
                    = sum[i * head_stride + j] / static_cast<scalar_t>(count[i]);
            }
        }
        else {
            label_counts[kv_head * nlist + i] = count[i];
        }
    }
}

template <int BLOCK_DIM>
__global__ void get_sel_indices_kernel(
    const int* neigh_c,
    const int* neigh_c_size,
    const int* cluster_size_ps,
    const int* cluster_key_indices,
    int* sel_indices,
    int num_heads,
    int num_kv_heads,
    int nlist,
    int budget,
    int seq_len
) {
    int tid = threadIdx.x;
    int block_size = blockDim.x;
    int head = blockIdx.x;  // Each block is processing a single kv_head
    int kv_head = head / (num_heads / num_kv_heads);

    extern __shared__ int sharedMem[];
    int* sh_neigh_size_ps = sharedMem;
    int* sh_num_need_c = &sharedMem[nlist];

    using BlockScan = cub::BlockScan<int, BLOCK_DIM>;
    __shared__ typename BlockScan::TempStorage cub_temp_storage;

    if (head < num_heads) {
        for (int i = tid; i < nlist; i += blockDim.x) {
            sh_neigh_size_ps[i] = neigh_c_size[head * nlist + i];
        }
        if (tid == 0) {
            *sh_num_need_c = nlist; // Initialize cutoff index to be out of bounds.
        }
        __syncthreads();

        int running_prefix = 0;
        // Process the array in tiles of size BLOCK_DIM
        for (int tile_offset = 0; tile_offset < nlist; tile_offset += block_size) {
            // Load this tile's data into registers
            int thread_data = 0;
            if (tile_offset + tid < nlist) {
                thread_data = sh_neigh_size_ps[tile_offset + tid];
            }

            // Scan the data in registers. The last thread gets the tile's total sum.
            int scan_output;
            BlockScan(cub_temp_storage).InclusiveSum(thread_data, scan_output);
            __syncthreads(); // Ensure scan is complete for all threads

            // Add the prefix from previous tiles and write back to shared memory
            if (tile_offset + tid < nlist) {
                sh_neigh_size_ps[tile_offset + tid] = scan_output + running_prefix;
            }
            __syncthreads(); // Ensure writes are complete before next tile

            // The last thread in the block holds the sum of the tile (in scan_output)
            // Broadcast this sum to all threads to update the running_prefix for the next tile.
            int tile_sum = __shfl_sync(0xFFFFFFFF, scan_output, block_size - 1);
            running_prefix += tile_sum;
        }

        // After prefix sum, find the index of the minimal element greater than threshold C
        for (int i = tid; i < nlist; i += block_size) {
            if (sh_neigh_size_ps[i] > budget) {
                atomicMin(sh_num_need_c, i+1);
            }
        }
        __syncthreads();

        for (int c = tid; c < *sh_num_need_c; c += block_size) {
            const int cid = neigh_c[head * nlist + c];
            const int ck_ind_end = cluster_size_ps[kv_head * nlist + cid];
            const int c_size = neigh_c_size[head * nlist + c];// or cluster_size[head*nlist + cid]
            const int sel_ind_end = sh_neigh_size_ps[c];

            for (int i = 1; i <= c_size; i++) {
                int read_idx = ck_ind_end - i;
                int store_idx = sel_ind_end - i;
                if (store_idx < budget) {
                    sel_indices[head * budget + store_idx] 
                        = cluster_key_indices[kv_head * seq_len + read_idx];
                }
            }
        }
    }
}

template <typename DistT>
__global__ void get_neigh_c_kernel(const DistT* q, const DistT* centroids, 
                                const int* cluster_size, 
                                int* neigh_c, int* neigh_c_size,
                                int nlist, int num_heads, int num_kv_heads, int head_dim) {
    int head_idx = blockIdx.x;  // One block for each kv_head
    int kv_head_idx = head_idx / (num_heads / num_kv_heads);
    if (head_idx >= num_heads) return;

    // Shared memory for sorting each row
    __shared__ DistT shared_dist[256];  // Adjust the size as needed
    __shared__ int shared_indices[256]; // Adjust the size as needed

    // Load data for the current head (row) into shared memory
    int idx = threadIdx.x;
    if (idx < nlist) {
        DistT dist_val = 0;
        for (int i = 0; i < head_dim; ++i) {
            dist_val += q[head_idx * head_dim + i] * centroids[kv_head_idx * nlist * head_dim + idx * head_dim + i];
        }
        shared_dist[idx] = dist_val;
        shared_indices[idx] = idx;
    }
    else {
        shared_dist[idx] = std::numeric_limits<DistT>::min();
        shared_indices[idx] = -1;
    }
    __syncthreads();

    // Perform sorting in shared memory (using Bitonic Sort)
    for (int size = 2; size <= nlist; size *= 2) {  // Iterate over increasing subsequence lengths
        for (int step = size / 2; step > 0; step /= 2) {
            int j = idx ^ step;  // Bitonic sort: compare elements with distance `step`
            if (j > idx && j < nlist) {  // Only compare valid indices
                // Swap if needed to ensure large-to-small order
                if (shared_dist[idx] < shared_dist[j]) {  
                    // Swap values in dist
                    DistT temp_dist = shared_dist[idx];
                    shared_dist[idx] = shared_dist[j];
                    shared_dist[j] = temp_dist;

                    // Swap the corresponding indices
                    int temp_idx = shared_indices[idx];
                    shared_indices[idx] = shared_indices[j];
                    shared_indices[j] = temp_idx;
                }
            }
            __syncthreads();  // Synchronize threads after each comparison
        }
    }

    // Write the sorted indices back to the global memory
    if (idx < nlist) {
        // if (head_idx == 0)
        //     printf("%d %f\n", idx, (float)shared_dist[idx]);
        int t = shared_indices[idx];
        neigh_c[head_idx * nlist + idx] = t;
        neigh_c_size[head_idx * nlist + idx] = cluster_size[kv_head_idx * nlist + t];
    }
}