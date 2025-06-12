#include "km_ops.h"
#include "pytorch_extension_utils.h"
#include "flashinfer/utils.cuh"


__global__ void
KMeansSearchIndicesKernel(int64_t* __restrict__ num_need_clusters,
						  int64_t* __restrict__ sel_cluster_size_ps,
						  int64_t* __restrict__ sel_cluster_key_start,
						  int64_t* __restrict__ sel_cluster_key_end,
						  int64_t* __restrict__ cluster_key_indices,
						  int64_t* __restrict__ sel_key_indices,
						  const int num_heads, const int num_kv_heads,
						  const size_t prefill_len,
						  const int max_num_need_clusters,
						  const size_t max_num_indices) {
	int head_idx = blockIdx.x;
	int num_kv_group = num_heads / num_kv_heads;
	int kv_head_idx = head_idx / num_kv_group;
	int cluster_idx = threadIdx.x;
	if (cluster_idx < num_need_clusters[head_idx]) {
		// Inversed order to use the prefix sum of torch.cumsum
		int64_t ind_start = sel_cluster_key_start[head_idx*max_num_need_clusters + cluster_idx];
		int64_t ind_end = sel_cluster_key_end[head_idx*max_num_need_clusters + cluster_idx];
		int64_t store_idx = sel_cluster_size_ps[head_idx*max_num_need_clusters + cluster_idx] - 1;
		for (int i = ind_end - 1; i >= ind_start; i--) {
			sel_key_indices[head_idx*max_num_indices + store_idx]
				= cluster_key_indices[kv_head_idx*prefill_len + i];
			store_idx--;
		}
	}
}

cudaError_t KMeansSearchIndices(int64_t* num_need_clusters,
								int64_t* sel_cluster_size_ps,
								int64_t* sel_cluster_key_start,
								int64_t* sel_cluster_key_end,
								int64_t* cluster_key_indices,
								int64_t* sel_key_indices,
								const int num_heads, const int num_kv_heads,
						  		const size_t prefill_len,
								const int max_num_need_clusters,
								const size_t max_num_indices,
								cudaStream_t stream = nullptr) {

	dim3 nblks(num_heads);
	dim3 nthrs(max_num_need_clusters);
	auto kernel = KMeansSearchIndicesKernel;
	void* args[] = {(void*)&num_need_clusters,
					(void*)&sel_cluster_size_ps,
					(void*)&sel_cluster_key_start,
					(void*)&sel_cluster_key_end,
					(void*)&cluster_key_indices,
					(void*)&sel_key_indices,
					(void*)&num_heads, (void*)&num_kv_heads,
					(void*)&prefill_len,
					(void*)&max_num_need_clusters,
					(void*)&max_num_indices};
	FLASHINFER_CUDA_CALL(cudaLaunchKernel((void*)kernel, nblks, nthrs, args, 0, stream));
	return cudaSuccess;
}

void search_indices(torch::Tensor num_need_clusters,
	 				torch::Tensor sel_cluster_size_ps,
					torch::Tensor sel_cluster_key_start,
					torch::Tensor sel_cluster_key_end,
					torch::Tensor cluster_key_indices,
					torch::Tensor sel_key_indices) {
#ifdef BSK_TORCH_CHECK
	CHECK_INPUT(num_need_clusters);		// [num_heads]
	CHECK_INPUT(sel_cluster_size_ps);	// [num_heads, max_num_need_clusters]
	CHECK_INPUT(sel_cluster_key_start);	// [num_heads, max_num_need_clusters]
	CHECK_INPUT(sel_cluster_key_end);	// [num_heads, max_num_need_clusters]
	CHECK_INPUT(cluster_key_indices);	// [num_kv_heads, prefill_len]
	CHECK_INPUT(sel_key_indices);		// [num_heads, max_num_indices]

	CHECK_EQ(num_need_clusters.size(0), sel_cluster_size_ps.size(0));
	CHECK_EQ(num_need_clusters.size(0), sel_cluster_key_start.size(0));
	CHECK_EQ(sel_cluster_key_start.size(0), sel_cluster_key_end.size(0));
	CHECK_EQ(sel_cluster_size_ps.size(1), sel_cluster_key_start.size(1));
	CHECK_EQ(sel_cluster_key_start.size(1), sel_cluster_key_end.size(1));
	CHECK_EQ(sel_cluster_key_start.size(0), sel_key_indices.size(0));

	CHECK_EQ(num_need_clusters.scalar_type(), torch::kInt64);
	CHECK_EQ(sel_cluster_size_ps.scalar_type(), torch::kInt64);
	CHECK_EQ(sel_cluster_key_start.scalar_type(), torch::kInt64);
	CHECK_EQ(sel_cluster_key_end.scalar_type(), torch::kInt64);
	CHECK_EQ(cluster_key_indices.scalar_type(), torch::kInt64);
	CHECK_EQ(sel_key_indices.scalar_type(), torch::kInt64);
#endif

	int num_heads = num_need_clusters.size(0);
	int num_kv_heads = cluster_key_indices.size(0);
	size_t prefill_len = cluster_key_indices.size(1);
	int max_num_need_clusters = sel_cluster_key_start.size(1);
	size_t max_num_indices = sel_key_indices.size(1);

	cudaError_t status = KMeansSearchIndices(
		static_cast<int64_t*>(num_need_clusters.data_ptr()),
		static_cast<int64_t*>(sel_cluster_size_ps.data_ptr()),
		static_cast<int64_t*>(sel_cluster_key_start.data_ptr()),
		static_cast<int64_t*>(sel_cluster_key_end.data_ptr()),
		static_cast<int64_t*>(cluster_key_indices.data_ptr()),
		static_cast<int64_t*>(sel_key_indices.data_ptr()),
		num_heads, num_kv_heads, 
		prefill_len,
		max_num_need_clusters,
		max_num_indices
	);

	TORCH_CHECK(status == cudaSuccess,
				"KMeans search indices failed",
				cudaGetErrorString(status));
}