
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

	def is_empty(self) -> bool:
		return self._counter == 0

	def get_mean(self) -> Optional[Tensor]:
		return self._sum / self._counter if self._sum is not None else None


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

	def is_empty(self) -> bool:
		return self._counter == 0

	def get_std(self) -> Optional[Tensor]:
		if self._items_sum is not None and self._items_sq_sum is not None:
			return torch.sqrt(self._items_sq_sum / self._counter - (self._items_sum / self._counter) ** 2)
		else:
			return None


class MinTracker(IncrementalMetric):
	def __init__(self):
		super().__init__()
		self._min = None

	def reset(self):
		self._min = None

	def add(self, value: Tensor):
		if self._min is None or self._min > value:
			self._min = value

	def get_current(self) -> Optional[Tensor]:
		return self.get_min()

	def is_empty(self) -> bool:
		return self._min is None

	def get_min(self) -> Optional[Tensor]:
		return self._min


class MaxTracker(IncrementalMetric):
	def __init__(self):
		super().__init__()
		self._max = None

	def reset(self):
		self._max = None

	def add(self, value: Tensor):
		if self._max is None or self._max < value:
			self._max = value

	def get_current(self) -> Optional[Tensor]:
		return self.get_max()

	def is_empty(self) -> bool:
		return self._max is None

	def get_max(self) -> Optional[Tensor]:
		return self._max
