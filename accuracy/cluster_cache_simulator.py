import torch
import queue, atexit
from datetime import datetime

class CacheSimulator():
	def __init__(self, layer_idx, num_step):
		self.layer_idx = layer_idx
		self.indices = queue.Queue(num_step)
		self.head_hit_rates = []
		self.step_hit_rates = []
		atexit.register(self.stat_hit_rate)

	def stat_hit_rate(self):
		if len(self.step_hit_rates) > 0:
			step_avg_hit_rate = sum(self.step_hit_rates) / len(self.step_hit_rates)
			print("Layer", self.layer_idx, "Overall hit rate:", step_avg_hit_rate)
			now = datetime.now().strftime("%y%m%d-%H%M")
			with open(f"cache_log/{now}.csv", "a") as log:
				log.write(f"{self.layer_idx},{step_avg_hit_rate}\n")
	
	def update_hit_rate(self):
		def get_hit_rate(previous: set, cur: list):
			cur_set = set(cur)
			assert len(cur_set) == len(cur)
			return len(cur_set & previous) / len(cur)

		cur_ind = self.indices.queue[-1]
		num_heads, num_sel_clusters = cur_ind.shape
		cur_ind = cur_ind.tolist()
		cached_cluster = [set() for _ in range(num_heads)]
		# Convert previous cluster indices into sets
		for i in range(self.indices.qsize() - 1):
			list_ind = self.indices.queue[i].tolist()
			for h in range(num_heads):
				cached_cluster[h].update(list_ind[h])
		head_hit_rates = []
		for h in range(num_heads):
			hr = get_hit_rate(cached_cluster[h], cur_ind[h])
			head_hit_rates.append(hr)
		head_avg_hit_rate = sum(head_hit_rates) / len(head_hit_rates)
		# print(head_avg_hit_rate)
		self.step_hit_rates.append(head_avg_hit_rate)

	def update(self, cluster_ind: torch.Tensor):
		self.indices.put(cluster_ind)
		if self.indices.full():
			self.update_hit_rate()
			# remove oldest element
			self.indices.get()
