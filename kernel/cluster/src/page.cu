#include "km_ops.h"
#include "pytorch_extension_utils.h"
#include "km_include/page.cuh"

using namespace flashinfer;

void append_kv_cache_prefill(torch::Tensor k,
							 torch::Tensor v,
							 torch::Tensor kv_data,
							 torch::Tensor kv_indices,
							 torch::Tensor kv_indptr,
							 unsigned int kv_last_page_len,
							 unsigned int kv_last_page_idx,
							 unsigned int layout) {
	constexpr size_t batch_size = 1;
	
#ifdef BSK_TORCH_CHECK
	CHECK_INPUT(k); // [bsz, num_kv_heads, head_dim]
	CHECK_INPUT(v); // [bsz, num_kv_heads, head_dim]
	// (num_max_pages, 2, H_kv, page_size, head_dim) for HND
	// (num_max_pages, 2, page_size, H_kv, head_dim) for NHD
	CHECK_INPUT(kv_data);
	CHECK_INPUT(kv_indices); // [num_pages]

	CHECK_DIM(1, kv_indices);
	CHECK_DIM(3, k);
	CHECK_DIM(3, v);
	CHECK_DIM(5, kv_data);

	CHECK_GE(k.size(0), 2); // Prefill
	CHECK_GE(v.size(0), 2); // Prefill
	CHECK_EQ(kv_indices.scalar_type(), torch::kInt32);
	CHECK_EQ(kv_indptr.scalar_type(), torch::kInt32);
#endif

	size_t seq_len = k.size(0);
	size_t num_kv_heads = k.size(1);
	size_t head_dim = k.size(2);
	size_t page_size;
	QKVLayout kv_layout = static_cast<QKVLayout>(layout);
	if(kv_layout == QKVLayout::kHND) {
		assert(false);
	} else {
		page_size = kv_data.size(2);
#ifdef BSK_TORCH_CHECK
		CHECK_EQ(kv_data.size(3), num_kv_heads);
		CHECK_EQ(kv_data.size(4), head_dim);
#endif
	}

#ifdef BSK_TORCH_CHECK
	CHECK_EQ(seq_len, v.size(0));
#endif

	torch::Tensor append_indptr =
		torch::tensor({0, static_cast<int32_t>(seq_len)}, kv_indices.options());

	bool success = DISPATCH_PYTORCH_DTYPE_TO_CTYPE(k.scalar_type(), c_type, [&] {
		SWITCH_LAYOUT(kv_layout, KV_LAYOUT, {
			paged_kv_t<PageStorage::kIndices, KV_LAYOUT, c_type, int32_t> paged_kv(
				num_kv_heads,
				page_size,
				head_dim,
				batch_size,
				0,
				kv_last_page_len,
				kv_last_page_idx,
				static_cast<c_type*>(kv_data.data_ptr()),
				static_cast<int32_t*>(kv_indices.data_ptr()),
				static_cast<int32_t*>(kv_indptr.data_ptr()));

			cudaError_t status =
				AppendPagedKVCachePrefill<PageStorage::kIndices, KV_LAYOUT, c_type, int32_t>(
					paged_kv,
					static_cast<c_type*>(k.data_ptr()),
					static_cast<c_type*>(v.data_ptr()),
					static_cast<int32_t*>(append_indptr.data_ptr()),
					nullptr);

			TORCH_CHECK(status == cudaSuccess,
						"Append_kv_cache_prefill failed with error code ",
						cudaGetErrorString(status));
		});
		return true;
	});

	TORCH_CHECK(success, "Append_kv_cache_prefill failed to dispatch with dtype ", k.scalar_type());
}

void append_kv_cache_decode(torch::Tensor k,
							torch::Tensor v,
							torch::Tensor kv_data,
							torch::Tensor kv_indices,
							torch::Tensor kv_indptr,
							unsigned int kv_last_page_len,
							unsigned int kv_last_page_idx,
							unsigned int layout) {
	constexpr size_t batch_size = 1;

#ifdef BSK_TORCH_CHECK
	CHECK_INPUT(k); // [bsz, num_kv_heads, head_dim]
	CHECK_INPUT(v); // [bsz, num_kv_heads, head_dim]
	// (num_max_pages, 2, H_kv, page_size, head_dim) for HND
	// (num_max_pages, 2, page_size, H_kv, head_dim) for NHD
	CHECK_INPUT(kv_data);
	CHECK_INPUT(kv_indices); // [num_pages]

	CHECK_DIM(1, kv_indices);
	CHECK_DIM(3, k);
	CHECK_DIM(3, v);
	CHECK_DIM(5, kv_data);

	CHECK_EQ(k.size(0), 1); // decode
	CHECK_EQ(v.size(0), 1); // decode
	CHECK_EQ(kv_indices.scalar_type(), torch::kInt32);
	CHECK_EQ(kv_indptr.scalar_type(), torch::kInt32);
#endif

	size_t num_kv_heads = k.size(1);
	size_t head_dim = k.size(2);
	size_t page_size;
	QKVLayout kv_layout = static_cast<QKVLayout>(layout);
	if(kv_layout == QKVLayout::kHND) {
		assert(false);
	} else {
		page_size = kv_data.size(2);
		CHECK_EQ(kv_data.size(3), num_kv_heads);
		CHECK_EQ(kv_data.size(4), head_dim);
	}
	
	bool success = DISPATCH_PYTORCH_DTYPE_TO_CTYPE(k.scalar_type(), c_type, [&] {
		SWITCH_LAYOUT(kv_layout, KV_LAYOUT, {
			paged_kv_t<PageStorage::kIndices, KV_LAYOUT, c_type, int32_t> paged_kv(
				num_kv_heads,
				page_size,
				head_dim,
				batch_size,
				0,
				kv_last_page_len,
				kv_last_page_idx,
				static_cast<c_type*>(kv_data.data_ptr()),
				static_cast<int32_t*>(kv_indices.data_ptr()),
				static_cast<int32_t*>(kv_indptr.data_ptr()));

			cudaError_t status =
				AppendPagedKVCacheDecode<PageStorage::kIndices, KV_LAYOUT, c_type, int32_t>(
					paged_kv,
					static_cast<c_type*>(k.data_ptr()),
					static_cast<c_type*>(v.data_ptr()),
					nullptr);

			TORCH_CHECK(status == cudaSuccess,
						"Append_kv_cache_decode failed with error code ",
						cudaGetErrorString(status));
		});
		return true;
	});

	TORCH_CHECK(success, "Append_kv_cache_decode failed to dispatch with dtype ", k.scalar_type());
}