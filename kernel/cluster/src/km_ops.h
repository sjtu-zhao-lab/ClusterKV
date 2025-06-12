#pragma once
#include <torch/extension.h>
#include "km_include/decode_handler.cuh"
#include <c10/cuda/CUDAStream.h>


void search_indices(torch::Tensor num_need_clusters,
	 				torch::Tensor sel_cluster_size_ps,
					torch::Tensor sel_cluster_key_start,
					torch::Tensor sel_cluster_key_end,
					torch::Tensor cluster_key_indices,
					torch::Tensor sel_key_indices);

void append_kv_cache_prefill(torch::Tensor k,
							 torch::Tensor v,
							 torch::Tensor kv_data,
							 torch::Tensor kv_indices,
							 torch::Tensor kv_indptr,
							 unsigned int kv_last_page_len,
							 unsigned int kv_last_page_idx,
							 unsigned int layout);

void append_kv_cache_decode(torch::Tensor k,
							torch::Tensor v,
							torch::Tensor kv_data,
							torch::Tensor kv_indices,
							torch::Tensor kv_indptr,
							unsigned int kv_last_page_len,
							unsigned int kv_last_page_idx,
							unsigned int layout);

torch::Tensor prefill_with_paged_kv_cache(torch::Tensor q,
										  torch::Tensor kv_data,
										  torch::Tensor kv_indices,
										  unsigned int kv_last_page_len,
										  bool causal,
										  unsigned int layout,
										  bool allow_fp16_qk_reduction,
										  float rope_scale,
										  float rope_theta);

class KMBatchDecodeWithPagedKVCachePyTorchWrapper {
public:
	static KMBatchDecodeWithPagedKVCachePyTorchWrapper Create(unsigned int layout) {
		return KMBatchDecodeWithPagedKVCachePyTorchWrapper(layout);
	}
	void BeginForward(torch::Tensor indptr,
					  unsigned int num_qo_heads,
					  unsigned int num_kv_heads,
					  unsigned int head_dim,
					  unsigned int page_size,
					  torch::Tensor empty_data);

	void EndForward();

	void Forward(torch::Tensor q,
				 torch::Tensor o,
				 torch::Tensor paged_kv_data,
				 torch::Tensor paged_kv_indices,
				 torch::Tensor paged_kv_indptr,
				 unsigned int paged_kv_last_page_len,
				 unsigned int paged_kv_last_page_idx,
				 float rope_scale,
				 float rope_theta);

	template<bool full>
	void ForwardImpl(torch::Tensor q,
					torch::Tensor o,
					torch::Tensor paged_kv_data,
					torch::Tensor paged_kv_indices,
					torch::Tensor paged_kv_indptr,
					unsigned int paged_kv_last_page_len,
					unsigned int paged_kv_last_page_idx,
					float rope_scale,
					float rope_theta);
private:
	KMBatchDecodeWithPagedKVCachePyTorchWrapper(unsigned int layout)
		: kv_layout_(flashinfer::QKVLayout(layout)) { }
	flashinfer::BatchDecodeHandler handler_;
	flashinfer::QKVLayout kv_layout_;
};

void update_centroids(torch::Tensor keys, torch::Tensor labels, torch::Tensor centroids, 
					int max_seq_len, int64_t cuda_stream);
void count_labels(torch::Tensor labels, torch::Tensor counts, 
				int max_seq_len, int64_t cuda_stream);

void get_sel_indices(torch::Tensor neigh_c, torch::Tensor neigh_c_size, 
					torch::Tensor cluster_size_ps, 
					torch::Tensor cluster_key_indices,
					torch::Tensor sel_indices);

void get_neigh_c(torch::Tensor q, torch::Tensor centroids, torch::Tensor cluster_size,
				torch::Tensor neigh_c, torch::Tensor neigh_c_size);

void recall(torch::Tensor gpu_cache, torch::Tensor cpu_cache,
			torch::Tensor topk_i, torch::Tensor g2c, torch::Tensor c2g,
			torch::Tensor is_in_cache, torch::Tensor is_in_topk, 
			torch::Tensor swap_out_indices, torch::Tensor swap_in_indices,
			torch::Tensor swap_out_count, torch::Tensor swap_in_count,
			int seq_len);