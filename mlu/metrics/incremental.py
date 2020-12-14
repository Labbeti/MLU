
import torch

from mlu.metrics.base import Metric, IncrementalMetric

from torch import Tensor
from typing import List, Optional


class IncrementalMean(IncrementalMetric):
	def __init__(self):
		"""
			Compute the continue average of a values.
		"""
		super().__init__()
		self._sum = None
		self._counter = 0

	def reset(self):
		self._sum = None
		self._counter = 0

	def add(self, value: Tensor):
		if self._sum is None:
			self._sum = value
			self._counter = 1
		else:
			self._sum += value
			self._counter += 1

	def get_current(self) -> Optional[Tensor]:
		return self.get_mean()

	def get_mean(self) -> Optional[Tensor]:
		return self._sum / self._counter if self._sum is not None else None

	def is_empty(self) -> bool:
		return self._counter == 0


class IncrementalStd(IncrementalMetric):
	def __init__(self):
		super().__init__()
		self._items_sum = None
		self._items_sq_sum = None
		self._counter = 0

	def reset(self):
		self._items_sum = None
		self._items_sq_sum = None
		self._counter = 0

	def add(self, value: Tensor):
		if isinstance(value, float):
			value = torch.scalar_tensor(value)

		if self._items_sum is None or self._items_sq_sum is None:
			self._items_sum = value
			self._items_sq_sum = value ** 2
			self._counter = 1
		else:
			self._items_sum += value
			self._items_sq_sum += value ** 2
			self._counter += 1

	def get_current(self) -> Optional[Tensor]:
		return self.get_std()

	def get_std(self) -> Optional[Tensor]:
		if self._items_sum is not None and self._items_sq_sum is not None:
			return torch.sqrt(self._items_sq_sum / self._counter - (self._items_sum / self._counter) ** 2)
		else:
			return None

	def is_empty(self) -> bool:
		return self._counter == 0


class IncrementalWrapper(Metric):
	"""
		Compute an incremental score (mean or std) of a metric.
	"""
	def __init__(self, metric: Metric, continue_metric: IncrementalMetric = IncrementalMean()):
		super().__init__()
		self.metric = metric
		self.continue_metric = continue_metric

	def compute_score(self, input_: Tensor, target: Tensor) -> Tensor:
		score = self.metric(input_, target)
		self.continue_metric.add(score)
		return self.continue_metric.get_current()


class IncrementalListWrapper(Metric):
	"""
		Compute a list of incremental scores (mean or std) of a metric.
	"""
	def __init__(self, metric: Metric, continue_metric_list: Optional[List[IncrementalMetric]] = None):
		super().__init__()
		self.metric = metric
		self.continue_metric_list = continue_metric_list if continue_metric_list is not None else [IncrementalMean()]

	def compute_score(self, input_: Tensor, target: Tensor) -> List[Tensor]:
		score = self.metric(input_, target)
		for continue_metric in self.continue_metric_list:
			continue_metric.add(score)
		return [continue_metric.get_current() for continue_metric in self.continue_metric_list]
