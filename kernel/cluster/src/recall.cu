#include "km_ops.h"
#include "pytorch_extension_utils.h"
#include "km_include/recall.cuh"

inline int next_pow2(int n) {
    if (n <= 0) return 1;
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n++;
    return n;
}

void recall(torch::Tensor gpu_cache, torch::Tensor cpu_cache,
			torch::Tensor topk_i, torch::Tensor g2c, torch::Tensor c2g,
			torch::Tensor is_in_cache, torch::Tensor is_in_topk, 
			torch::Tensor swap_out_indices, torch::Tensor swap_in_indices,
			torch::Tensor swap_out_count, torch::Tensor swap_in_count,
			int seq_len) {
#ifdef BSK_TORCH_CHECK
	CHECK_DIM(4, gpu_cache);			// [budget-1, 2, num_heads, head_dim]
	CHECK_DIM(4, cpu_cache);			// [max_seqlen, 2, num_kv_heads, head_dim]
	CHECK_DIM(2, topk_i);				// [num_heads, budget-1]
	CHECK_DIM(2, c2g);					// [num_heads, max_seqlen]
	CHECK_DIM(2, g2c);					// [num_heads, budget-1]
	CHECK_DIM(2, is_in_cache);			// [num_heads, max_seqlen]
	CHECK_DIM(2, is_in_topk);			// [num_heads, max_seqlen]
	CHECK_DIM(2, swap_out_indices);		// [num_heads, budget-1]
	CHECK_DIM(2, swap_in_indices);		// [num_heads, budget-1]
	CHECK_DIM(1, swap_out_count);		// [num_heads]
	CHECK_DIM(1, swap_in_count);		// [num_heads]
	CHECK_INPUT(gpu_cache);
	CHECK_CONTIGUOUS(cpu_cache);
	CHECK_INPUT(topk_i);
	CHECK_INPUT(c2g);
	CHECK_INPUT(g2c);
	CHECK_EQ(topk_i.scalar_type(), torch::kInt32);
	CHECK_EQ(g2c.scalar_type(), torch::kInt32);
	CHECK_EQ(c2g.scalar_type(), torch::kInt32);
	CHECK_EQ(is_in_cache.scalar_type(), torch::kBool);
	CHECK_EQ(is_in_topk.scalar_type(), torch::kBool);
	CHECK_EQ(swap_out_indices.scalar_type(), torch::kInt32);
	CHECK_EQ(swap_in_indices.scalar_type(), torch::kInt32);
	CHECK_EQ(swap_out_count.scalar_type(), torch::kInt32);
	CHECK_EQ(swap_in_count.scalar_type(), torch::kInt32);
	CHECK_EQ(topk_i.size(1), swap_out_indices.size(1));
	CHECK_EQ(topk_i.size(1), swap_in_indices.size(1));
#endif
	int num_heads = topk_i.size(0);
	int num_kv_heads = cpu_cache.size(2);
	int group_size = num_heads / num_kv_heads;
	int budget = topk_i.size(1); // budget-1
	int head_dim = gpu_cache.size(3);
	int max_seqlen = cpu_cache.size(0);
	int gpu_budget = gpu_cache.size(0); // should be budget-1
	int gpu_seq_stride = gpu_cache.stride(0);
	int gpu_kv_stride = gpu_cache.stride(1);
	int cpu_seq_stride = cpu_cache.stride(0);
	int cpu_kv_stride = cpu_cache.stride(1);

	int32_t* topk_i_ptr = static_cast<int32_t*>(topk_i.data_ptr());
	int32_t* c2g_ptr = static_cast<int32_t*>(c2g.data_ptr());
	int32_t* g2c_ptr = static_cast<int32_t*>(g2c.data_ptr());
	
	const int gridSize = num_heads;
	const int blockSize = next_pow2(budget) > 1024 ? 1024 : next_pow2(budget);
	const size_t shmem = (2 * budget + 2) * sizeof(int);
	swap_in_out_kernel
	<<<gridSize, blockSize, shmem>>>(
		g2c_ptr, topk_i_ptr, c2g_ptr,
		static_cast<bool*>(is_in_cache.data_ptr()),
		static_cast<bool*>(is_in_topk.data_ptr()),
		static_cast<int*>(swap_out_indices.data_ptr()),
		static_cast<int*>(swap_in_indices.data_ptr()),
		static_cast<int*>(swap_out_count.data_ptr()),
		static_cast<int*>(swap_in_count.data_ptr()),
		budget, seq_len, max_seqlen
	);
	cudaError_t err = cudaGetLastError();
	TORCH_CHECK(err == cudaSuccess, "swap_in_out_kernel failed: ", cudaGetErrorString(err));

	void *cpu_buffer_mapped_ptr_void = nullptr;
    cudaError_t err1 = cudaHostGetDevicePointer(
		&cpu_buffer_mapped_ptr_void, 
		const_cast<void *>(cpu_cache.data_ptr()), // Needs non-const but we won't write
		0);
    TORCH_CHECK(err1 == cudaSuccess, 
      "cudaHostGetDevicePointer failed! Ensure CPU buffer is pinned and driver/runtime supports mapping. Error: ", 
      cudaGetErrorString(err));

	dim3 blockSize1(512 / head_dim, head_dim);
	DISPATCH_PYTORCH_DTYPE_TO_CTYPE(gpu_cache.scalar_type(), c_type, [&] {
		copy_kernel<c_type>
		<<<gridSize, blockSize1>>>(
			static_cast<c_type*>(gpu_cache.data_ptr()),
			static_cast<c_type*>(cpu_buffer_mapped_ptr_void),
			static_cast<int*>(swap_out_indices.data_ptr()),
			static_cast<int*>(swap_in_indices.data_ptr()),
			static_cast<int*>(swap_out_count.data_ptr()),
			group_size, budget, gpu_seq_stride, gpu_kv_stride,
			cpu_seq_stride, cpu_kv_stride
		);
		cudaError_t err = cudaGetLastError();
		TORCH_CHECK(err == cudaSuccess, "copy_kernel failed: ", cudaGetErrorString(err));
		return true;
	});
}