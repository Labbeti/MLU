
import torch

from mlu.metrics.base import Metric, IncrementalMetric

from torch import Tensor
from typing import List, Optional


class IncrementalMean(IncrementalMetric):
	def __init__(self):
		super().__init__()
		self._sum = None
		self._counter = 0

	def reset(self):
		self._sum = None
		self._counter = 0

	def add(self, item: Tensor):
		if self._sum is None:
			self._sum = item
			self._counter = 1
		else:
			self._sum += item
			self._counter += 1

	def get_current(self) -> Optional[Tensor]:
		return self.get_mean()

	def get_mean(self) -> Optional[Tensor]:
		if self._sum is not None:
			return self._sum / self._counter
		else:
			return None


class IncrementalStd(IncrementalMetric):
	def __init__(self):
		super().__init__()
		self._items = []

	def reset(self):
		self._items = []

	def add(self, item: Tensor):
		if isinstance(item, float):
			item = torch.scalar_tensor(item)
		self._items.append(item)

	def get_current(self) -> Optional[Tensor]:
		return self.get_std()

	def get_std(self) -> Optional[Tensor]:
		return torch.stack(self._items).std() if len(self._items) > 0 else None


class IncrementalWrapper(Metric):
	"""
		Compute the continue average of a metric.
	"""
	def __init__(self, metric: Metric, continue_metric: IncrementalMetric = IncrementalMean()):
		super().__init__()
		self.metric = metric
		self.continue_metric = continue_metric

	def compute_score(self, input_: Tensor, target: Tensor) -> Tensor:
		item = self.metric(input_, target)
		self.continue_metric.add(item)
		return self.continue_metric.get_current()


class IncrementalListWrapper(Metric):
	"""
		Compute the continue average of a metric.
	"""
	def __init__(self, metric: Metric, continue_metric_list: Optional[List[IncrementalMetric]] = None):
		super().__init__()
		self.metric = metric
		self.continue_metric_list = continue_metric_list if continue_metric_list is not None else [IncrementalMean()]

	def compute_score(self, input_: Tensor, target: Tensor) -> List[Tensor]:
		item = self.metric(input_, target)
		for continue_metric in self.continue_metric_list:
			continue_metric.add(item)
		return [continue_metric.get_current() for continue_metric in self.continue_metric_list]
