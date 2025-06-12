#include <torch/extension.h>
#include "km_ops.h"

PYBIND11_MODULE(_clusterkv_knl, m) {
	m.def("search_indices", &search_indices, "search kv incides of needed clusters");
	m.def("append_kv_cache_prefill", &append_kv_cache_prefill, "Append KV-Cache Prefill operator");
	m.def("append_kv_cache_decode", &append_kv_cache_decode, "Append KV-Cache Decode operator");
	m.def("prefill_with_paged_kv_cache",
		  &prefill_with_paged_kv_cache,
		  "Multi-request batch prefill with paged KV-Cache operator");
	py::class_<KMBatchDecodeWithPagedKVCachePyTorchWrapper>(
		m, "KMBatchDecodeWithPagedKVCachePyTorchWrapper")
		.def(py::init(&KMBatchDecodeWithPagedKVCachePyTorchWrapper::Create))
		.def("begin_forward", &KMBatchDecodeWithPagedKVCachePyTorchWrapper::BeginForward)
		.def("end_forward", &KMBatchDecodeWithPagedKVCachePyTorchWrapper::EndForward)
		.def("forward", &KMBatchDecodeWithPagedKVCachePyTorchWrapper::Forward);
	m.def("update_centroids", &update_centroids, "Build KMeans clusters and predict");
	m.def("count_labels", &count_labels, "Build KMeans clusters and predict");
	m.def("get_sel_indices", &get_sel_indices, "Build KMeans clusters and predict");
	m.def("get_neigh_c", &get_neigh_c, "Build KMeans clusters and predict");
	m.def("recall", &recall, "Build KMeans clusters and predict");
}