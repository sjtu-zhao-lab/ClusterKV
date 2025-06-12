#include "km_ops.h"
#include "pytorch_extension_utils.h"
#include "km_include/cluster.cuh"
#include <cutlass/half.h>

void update_centroids(torch::Tensor keys, 
					torch::Tensor labels, 
					torch::Tensor centroids,
					int max_seq_len,
					int64_t cuda_stream) {
#ifdef BSK_TORCH_CHECK
	CHECK_DIM(3, keys);			// [num_kv_heads, seq_len, head_dim]
	CHECK_DIM(2, labels);		// [num_kv_heads, seq_len (max_seq_len)]
	CHECK_DIM(3, centroids);	// [num_kv_heads, nlist, head_dim]
	CHECK_INPUT(keys);
	CHECK_INPUT(centroids);
	CHECK_EQ(keys.scalar_type(), centroids.scalar_type());
#endif
	// For llama-2 and llama-3 on 4090
	constexpr int channel_split = 8;
	const int num_block_threads = 512;

	const int num_kv_heads = keys.size(0);
    const int seq_len = keys.size(1);
    const int head_dim = keys.size(2);
    const int nlist = centroids.size(1);
	TORCH_CHECK(nlist < 180 * channel_split, "exceed max nlist allowed (shmem)");

	const int head_stride = head_dim / channel_split;
    const dim3 gridSize(num_kv_heads, channel_split);
	const dim3 blockSize(head_stride, num_block_threads / head_stride);
	cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
	bool success = DISPATCH_PYTORCH_DTYPE_TO_CTYPE(keys.scalar_type(), c_type, [&] {
		const int shmemUse = (nlist * head_stride) * sizeof(c_type) + nlist * sizeof(int);
		// printf("Shmem use %d, half size %d\n", shmemUse, sizeof(c_type));
		update_centroids_kernel<c_type, channel_split, false>
		<<<gridSize, blockSize, shmemUse, stream>>>(
			static_cast<c_type*>(keys.data_ptr()),
			static_cast<int*>(labels.data_ptr()),
			static_cast<c_type*>(centroids.data_ptr()),
			nullptr,
			num_kv_heads,
			seq_len,
			head_dim,
			nlist,
			max_seq_len
		);
		cudaError_t err = cudaGetLastError();
		TORCH_CHECK(err == cudaSuccess, "update_centroids_kernel failed", cudaGetErrorString(err));
		return true;
	});
}

void count_labels(torch::Tensor labels, 
				torch::Tensor counts, 
				int max_seq_len,
				int64_t cuda_stream) {
#ifdef BSK_TORCH_CHECK
	CHECK_DIM(2, labels);		// [num_kv_heads, seq_len (max_seq_len)]
	CHECK_DIM(2, counts);		// [num_kv_heads, nlist]
	CHECK_INPUT(counts);
	CHECK_EQ(labels.scalar_type(), torch::kInt32);
	CHECK_EQ(counts.scalar_type(), torch::kInt32);
#endif
	const int num_kv_heads = labels.size(0);
    const int seq_len = labels.size(1);
    const int nlist = counts.size(1);

    const int gridSize = num_kv_heads;
	// For llama-2 and llama-3 on 4090
	const int num_block_threads = 512;
	constexpr int channel_split = 1;
	const dim3 blockSize(channel_split, num_block_threads / channel_split);
	const int shmemUse = nlist * sizeof(int);
	cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
	update_centroids_kernel<int*, channel_split, true>
	<<<gridSize, blockSize, shmemUse, stream>>>(
		nullptr,
		static_cast<int*>(labels.data_ptr()),
		nullptr,
		static_cast<int*>(counts.data_ptr()),
		num_kv_heads,
		seq_len,
		0,
		nlist,
		max_seq_len
	);
	cudaError_t err = cudaGetLastError();
	TORCH_CHECK(err == cudaSuccess, "update_centroids_kernel failed", cudaGetErrorString(err));
}

void get_neigh_c(torch::Tensor q, torch::Tensor centroids, torch::Tensor cluster_size,
				torch::Tensor neigh_c, torch::Tensor neigh_c_size) {
#ifdef BSK_TORCH_CHECK
	CHECK_INPUT(q);					// [1, num_heads, head_dim]
	CHECK_INPUT(centroids);			// [num_kv_heads, nlist, head_dim]
	CHECK_INPUT(cluster_size);		// [num_kv_heads, nlist]
	CHECK_DIM(2, cluster_size);		
	CHECK_EQ(neigh_c.scalar_type(), torch::kInt32);
	CHECK_EQ(neigh_c_size.scalar_type(), torch::kInt32);
#endif
    const int num_heads = q.size(1);
    const int num_kv_heads = centroids.size(0);
    const int nlist = cluster_size.size(1);
	const int head_dim = centroids.size(2);

    // Define block and grid sizes
    const int block_size = 256;
    const int grid_size = num_heads;

    // Launch the kernel
	bool success = DISPATCH_PYTORCH_DTYPE_TO_CTYPE(q.scalar_type(), c_type, [&] {
		get_neigh_c_kernel<<<grid_size, block_size>>>(
			static_cast<c_type*>(q.data_ptr()),
			static_cast<c_type*>(centroids.data_ptr()),
			static_cast<int*>(cluster_size.data_ptr()),
			static_cast<int*>(neigh_c.data_ptr()),
			static_cast<int*>(neigh_c_size.data_ptr()),
			nlist, num_heads, num_kv_heads, head_dim);
		cudaError_t err = cudaGetLastError();
		TORCH_CHECK(err == cudaSuccess, "get_neigh_c_kernel failed", cudaGetErrorString(err));
		return true;
	});
}

void get_sel_indices(torch::Tensor neigh_c, torch::Tensor neigh_c_size, 
					torch::Tensor cluster_size_ps, 
					torch::Tensor cluster_key_indices,
					torch::Tensor sel_indices) {
#ifdef BSK_TORCH_CHECK
	CHECK_DIM(2, neigh_c);				// [num_heads, nlist]
	CHECK_DIM(2, neigh_c_size);			// [num_heads, nlist]
	CHECK_DIM(2, cluster_size_ps);		// [num_kv_heads, nlist]
	CHECK_DIM(2, cluster_key_indices);	// [num_kv_heads, kv_seq_len]
	CHECK_DIM(2, sel_indices);			// [num_heads, budget]
	CHECK_INPUT(neigh_c);
	CHECK_INPUT(neigh_c_size);
	CHECK_INPUT(cluster_size_ps);
	CHECK_INPUT(sel_indices);
	CHECK_EQ(neigh_c.scalar_type(), torch::kInt32);
	CHECK_EQ(neigh_c_size.scalar_type(), torch::kInt32);
	CHECK_EQ(cluster_size_ps.scalar_type(), torch::kInt32);
	CHECK_EQ(cluster_key_indices.scalar_type(), torch::kInt32);
	CHECK_EQ(sel_indices.scalar_type(), torch::kInt32);
#endif
	int num_heads = neigh_c.size(0);
	int nlist = neigh_c.size(1);
	int num_kv_heads = cluster_size_ps.size(0);
	int budget = sel_indices.size(1);
	int seq_len = cluster_key_indices.size(1);

	const int gridSize = num_heads;
	// const int num_block_threads = ;
	constexpr int blockSize = 256;
	const int shmemUse = (nlist + 1) * sizeof(int);
	get_sel_indices_kernel<blockSize>
	<<<gridSize, blockSize, shmemUse>>>(
		static_cast<int*>(neigh_c.data_ptr()),
		static_cast<int*>(neigh_c_size.data_ptr()),
		static_cast<int*>(cluster_size_ps.data_ptr()),
		static_cast<int*>(cluster_key_indices.data_ptr()),
		static_cast<int*>(sel_indices.data_ptr()),
		num_heads, num_kv_heads, nlist, budget, seq_len
	);
	cudaError_t err = cudaGetLastError();
	TORCH_CHECK(err == cudaSuccess, "get_sel_indices_kernel failed: ", cudaGetErrorString(err));
}