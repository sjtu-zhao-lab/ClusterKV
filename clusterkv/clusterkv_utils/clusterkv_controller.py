from clusterkv.utils.utils import TensorLayout
from clusterkv.clusterkv_utils.decode_wrapper import BatchDecodeWithPagedKVCacheWrapper
from transformers.models.llama.modeling_llama import repeat_kv

import torch
from typing import Tuple, List, Optional

class ClusterKVController:
	def __init__(
		self,
		num_layers,
		num_heads,
		num_kv_heads,
		head_dim,
		nlist,
		niter,
		token_budget,
		max_seq_len, # Real max for allocating kv / metadata
		dtype,
		device,      
		full,
		sink,
		window,
		window_nlist,
		offload
	):
		self.full = full
		# page_size = 1
		self.num_layers = num_layers
		self.num_heads = num_heads
		self.num_kv_heads = num_kv_heads
		self.num_key_value_groups = num_heads // num_kv_heads
		self.head_dim = head_dim
		self.kv_head_slice = torch.arange(0, num_heads, self.num_key_value_groups, device=device)
		self.dtype = dtype
		self.device = device
		self.layout = TensorLayout.NHD # Arbitrarily choose NHD. 

		self.max_seq_len = max_seq_len
		self.kv_cache: List[torch.Tensor] = [None] * num_layers
		for i in range(2):
			self.kv_cache[i] = torch.empty((max_seq_len, 2, 1, num_kv_heads, head_dim),
											dtype=dtype, device=device)
		self.num_kv_heads_ = num_heads if offload else num_kv_heads
		kv_cache_size = max_seq_len if not offload else sink + token_budget + window
		for i in range(num_layers - 2):
			self.kv_cache[2+i] = torch.empty((kv_cache_size, 2, 1, self.num_kv_heads_, head_dim),
											dtype=dtype, device=device)
		
		# ==================================== Offload related ====================================
		self.default_stream = torch.cuda.default_stream()
		self.offload = offload
		self.offload_stream = None
		self.offload_events = None
		self.kv_cache_cpu: List[torch.Tensor] = [None] * num_layers
		self.g2c: List[torch.Tensor] = [None] * num_layers
		self.c2g: Optional[torch.Tensor] = None
		self.is_in_cache: Optional[torch.Tensor] = None
		self.is_in_topk: Optional[torch.Tensor] = None
		self.swap_out_indices: Optional[torch.Tensor] = None
		self.swap_in_indices: Optional[torch.Tensor] = None
		self.swap_out_count: Optional[torch.Tensor] = None
		self.swap_in_count: Optional[torch.Tensor] = None
		assert not (offload and full), "offload cannot be enabled for full kv"
		if offload:
			for i in range(num_layers - 2):
				self.kv_cache_cpu[2+i] = torch.empty((max_seq_len, 2, num_kv_heads, head_dim),
												dtype=dtype, device="cpu", pin_memory=True)
				# init index after prefill, the g do not include sink
				t_range = torch.arange(0, token_budget-1, dtype=torch.int32, device=device).repeat(num_heads, 1)
				self.g2c[2+i] = torch.full(
					(num_heads, token_budget-1), -1, dtype=torch.int32, device=device
				)
				self.g2c[2+i].copy_(t_range)
				self.g2c[2+i] += sink
			self.offload_stream = torch.cuda.Stream(device)
			self.offload_events = [torch.cuda.Event() for _ in range(num_layers)]

			self.c2g = torch.zeros((num_heads, max_seq_len), dtype=torch.int32, device=device)
			self.is_in_cache = torch.zeros((num_heads, max_seq_len), dtype=torch.bool, device=device)
			self.is_in_topk = torch.zeros((num_heads, max_seq_len), dtype=torch.bool, device=device)
			self.swap_out_indices = torch.zeros((num_heads, token_budget-1), dtype=torch.int32, device=device)
			self.swap_in_indices = torch.zeros((num_heads, token_budget-1), dtype=torch.int32, device=device)
			self.swap_out_count = torch.zeros(num_heads, dtype=torch.int32, device=device)
			self.swap_in_count = torch.zeros(num_heads, dtype=torch.int32, device=device)	

		self.kv_indptr_for_append_offload = None
		self.kv_indices_with_last_offload = None
		self.kv_last_page_idx_offload = None # For decoding self-attention

		# ==================================== Offload related ====================================
			
		self.prompt_len = 0

		self.kv_seqlen = 0
		# self.kv_indices = []
		self.all_kv_indices = torch.arange(max_seq_len, dtype=torch.int32, device=device)
		self.empty_data = torch.empty(1, device=device)

		self.sink = sink
		self.sink_indices = torch.arange(self.sink, dtype=torch.int32, device=device).repeat(num_heads, 1)
		self.window = window
		self.win_indices = torch.arange(self.window, dtype=torch.int32, device=device).repeat(num_heads, 1)
		self.window_nlist = window_nlist

		self.nlist = nlist
		self.niter = niter
		self.labels = torch.ones(
			(num_kv_heads, max_seq_len), 
			dtype=torch.int32, device=device) * -1
		self.centroids = [None for _ in range(num_layers)]
		self.cluster_size = [None for _ in range(num_layers)]
		self.cluster_size_ps = [None for _ in range(num_layers)]
		self.cluster_key_indices = [None for _ in range(num_layers)]

		self.sel_token_indices = torch.full(
			(num_heads, token_budget-1), max_seq_len+1,
			dtype=torch.int32, device=device)
		self.neigh_c = torch.zeros(
			(num_heads, nlist), dtype=torch.int32, device=device)
		self.neigh_c_size = torch.zeros(
			(num_heads, nlist), dtype=torch.int32, device=device)

		self._token_budget = token_budget
		self.infer_token_budget = None
		self._max_page_limit = 1024*1024 # arbitraty large size

		self.kv_indptr_for_append = None
		self.kv_indices_with_last = None
		self.kv_last_page_idx = None # For decoding self-attention
		self.last_page_len = 1

		self._decode_handler = BatchDecodeWithPagedKVCacheWrapper(kv_layout="NHD")

		self.overlap_build = True
		self.build_cluster_stream = None
		self.build_cluster_events = [None] * num_layers
		self.build_cluster_finish = None
		if self.overlap_build:
			self.build_cluster_stream = torch.cuda.Stream(device)
			self.build_cluster_events = [torch.cuda.Event() for _ in range(num_layers)]
			self.build_cluster_finish = [False] * num_layers
	
	def prepare_metadata(self, seq_len: int):
		self.kv_seqlen += seq_len
		if seq_len > 1:
			self.prompt_len = seq_len
			self.win_indices += self.prompt_len
		elif self.generated_len > self.window and self.generated_len % self.window == 1:
			self.win_indices += self.window
	
	@property
	def generated_len(self):
		return self.kv_seqlen - self.prompt_len

	@property
	def cur_win_size(self):
		if self.generated_len == 0:
			return 0
		return (self.generated_len % self.window) or self.window

	@property
	def cur_win_indices(self):
		return self.win_indices[:, :self.cur_win_size]

	def kv_cache_mid(self, layer_idx):
		return self.kv_cache[layer_idx][self.sink: self.sink+self._token_budget-1, :, 0, ...]

	def begin_forward(self, seq_len: int, updateTensor: bool = True):
		torch.cuda.nvtx.range_push("begin_forward")
		if updateTensor:
			self.kv_indptr_for_append = torch.tensor([0, self.kv_seqlen], 
													dtype=torch.int32, device=self.device)
			self.kv_last_page_idx = self.kv_seqlen - 1
			self.kv_indices_with_last = self.all_kv_indices[:self.kv_seqlen]
		if self.offload:
			offload_seqlen = self.sink + self._token_budget + self.cur_win_size
			# print(offload_seqlen, self._token_budget, self.generated_len)
			self.kv_indptr_for_append_offload = torch.tensor([0, offload_seqlen], 
													dtype=torch.int32, device=self.device)
			self.kv_indices_with_last_offload = self.all_kv_indices[:offload_seqlen]
			self.kv_last_page_idx_offload = offload_seqlen - 1
		if seq_len > 1:
			# prefill requests
			pass
		else:
			# decode requests
			self.infer_token_budget = min(self.sink + self._token_budget, self.kv_seqlen)
			cur_win_size = 0 if self._token_budget > self.kv_seqlen else self.cur_win_size	# full or no offload
			self.kv_indptr_for_approx_decode = torch.tensor([0, self.infer_token_budget + cur_win_size], 
															dtype=torch.int32, device=self.device)
			self._decode_handler.begin_forward(
				self.kv_indptr_for_approx_decode,
				self.num_heads,
				self.num_kv_heads if updateTensor else self.num_kv_heads_,
				self.head_dim,
				1,
				self.dtype
			)
		torch.cuda.nvtx.range_pop()
		
	def end_forward(self):
		self._decode_handler.end_forward()
	
	def get_k(self, layer_idx) -> torch.Tensor:
		return self.kv_cache[layer_idx][:self.kv_seqlen, 0, ...]

	def get_kv(self, layer_idx) -> Tuple[torch.Tensor, torch.Tensor]:
		return self.kv_cache[layer_idx][:self.kv_seqlen, 0, ...], \
				self.kv_cache[layer_idx][:self.kv_seqlen, 1, ...]

	def get_app_k_clustering(self, layer_idx) -> torch.Tensor:
		if self.offload:
			return self.kv_cache[layer_idx][-self.window:, 0, 0, self.kv_head_slice, :]
		else:
			return self.kv_cache[layer_idx][self.kv_seqlen-self.window: self.kv_seqlen, 0, 0, ...]

	def get_centroids(self, layer_idx) -> torch.Tensor:
		return self.centroids[layer_idx]

	def get_labels(self, seq_len) -> torch.Tensor:
		return self.labels[:, :seq_len]

	def get_cluster_size(self, layer_idx) -> torch.Tensor:
		return self.cluster_size[layer_idx]

	def get_cluster_size_ps(self, layer_idx) -> torch.Tensor:
		return self.cluster_size_ps[layer_idx]

	def get_cluster_key_indices(self, layer_idx) -> torch.Tensor:
		return self.cluster_key_indices[layer_idx][:, :self.kv_seqlen]
	
	def get_update_sel_metadata(self, layer_idx):
		return self.centroids[layer_idx], self.cluster_size[layer_idx], \
			self.cluster_size_ps[layer_idx], \
			self.cluster_key_indices[layer_idx][:, :self.kv_seqlen], \
			self.neigh_c, self.neigh_c_size

	def need_estimate(self) -> bool:
		if self.infer_token_budget is None:
			return False
		return self.kv_seqlen > self.infer_token_budget

	def clean_states(self):
		self.prompt_len = 0
		self.kv_seqlen = 0
		self.win_indices = torch.arange(self.window, dtype=torch.int32, device=self.device).repeat(self.num_heads, 1)
		self.labels.fill_(-1)
		self.centroids = [None for _ in range(self.num_layers)]
		self.cluster_size = [None for _ in range(self.num_layers)]
		self.cluster_size_ps = [None for _ in range(self.num_layers)]
		self.cluster_key_indices = [None for _ in range(self.num_layers)]
		self.sel_token_indices.fill_(self.max_seq_len+1)
		self.neigh_c.fill_(0)
		self.neigh_c_size.fill_(0)
		if self.overlap_build:
			self.build_cluster_stream = torch.cuda.Stream(self.device)
			self.build_cluster_events = [torch.cuda.Event() for _ in range(self.num_layers)]
			self.build_cluster_finish = [False] * self.num_layers
	
	def set_token_budget(self, token_budget):
		self._token_budget = token_budget
	
	def update_metadata(
		self, layer_idx, 
		centroids, 							# [num_kv_heads, nlist, head_dim]
		cluster_size, cluster_size_ps,		# [num_kv_heads, nlist]
		cluster_key_indices					# [num_kv_heads, seq_len]
	):
		if self.centroids[layer_idx] is None:
			# first clustering during prefill
			self.centroids[layer_idx] = centroids
			self.cluster_size[layer_idx] = cluster_size
			self.cluster_size_ps[layer_idx] = cluster_size_ps.to(torch.int32)
			self.cluster_key_indices[layer_idx] = cluster_key_indices.to(torch.int32)
		else:
			# append clustering during decoding
			append_nlist = centroids.shape[1]
			self.centroids[layer_idx] = torch.cat(
				[self.centroids[layer_idx], centroids], dim=1
			).contiguous()
			self.cluster_size[layer_idx] = torch.cat(
				[self.cluster_size[layer_idx], cluster_size], dim=1
			).contiguous()
			cluster_size_ps += self.cluster_size_ps[layer_idx][0, -1]
			self.cluster_size_ps[layer_idx] = torch.cat(
				[self.cluster_size_ps[layer_idx], cluster_size_ps.to(torch.int32)], dim=1
			).contiguous()
			self.cluster_key_indices[layer_idx] = torch.cat(
				[self.cluster_key_indices[layer_idx], cluster_key_indices.to(torch.int32)], dim=1
			).contiguous()
			if layer_idx == 2:
				self.neigh_c = torch.cat([self.neigh_c, 
					torch.zeros((self.num_heads, append_nlist), dtype=torch.int32, device=self.device),
				], dim=1).contiguous()
				self.neigh_c_size = torch.cat([self.neigh_c_size, 
					torch.zeros((self.num_heads, append_nlist), dtype=torch.int32, device=self.device)
				], dim=1).contiguous()
	
	def offload_prefill_kv(self, layer_idx, k, v):
		seq_len, num_kv_heads, head_dim = k.shape
		with torch.cuda.stream(self.offload_stream):
			self.kv_cache_cpu[layer_idx][:seq_len, 0, ...].copy_(k, non_blocking=True)
			self.kv_cache_cpu[layer_idx][:seq_len, 1, ...].copy_(v, non_blocking=True)
			self.offload_events[layer_idx].record(self.offload_stream)

	def offload_window_kv(self, layer_idx):
		window = self.window
		kv = self.kv_cache[layer_idx][-window:, :, 0, self.kv_head_slice, :]
		with torch.cuda.stream(self.offload_stream):
			self.kv_cache_cpu[layer_idx][self.kv_seqlen-window: self.kv_seqlen, ...].copy_(
				kv, non_blocking=True
			)
			self.offload_events[layer_idx].record(self.offload_stream)