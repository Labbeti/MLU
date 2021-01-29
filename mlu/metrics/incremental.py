
import torch

from mlu.metrics.base import IncrementalMetric

from torch import Tensor
from typing import Optional


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
		if not isinstance(value, Tensor):
			value = torch.as_tensor(value)

		if self._sum is None:
			self._sum = value.clone()
			self._counter = 1
		else:
			self._sum += value
			self._counter += 1

	def is_empty(self) -> bool:
		return self._counter == 0

	def get_current(self) -> Optional[Tensor]:
		return self.get_mean()

	def get_mean(self) -> Optional[Tensor]:
		return (self._sum / self._counter) if self._counter > 0 else None

	def get_nb_values_added(self) -> int:
		return self._counter


class IncrementalStd(IncrementalMetric):
	def __init__(self, unbiased: bool = False):
		"""
			Compute the continue unbiased Standard Deviation (std).

			:param unbiased: If True, apply the bessel correction to std (like in the default std of pytorch).
				Otherwise return the classic std (like the default std of numpy). (default: False)
		"""
		super().__init__()
		self._unbiased = unbiased
		self._items_sum = None
		self._items_sq_sum = None
		self._counter = 0

	def reset(self):
		self._items_sum = None
		self._items_sq_sum = None
		self._counter = 0

	def add(self, value: Tensor):
		if not isinstance(value, Tensor):
			value = torch.as_tensor(value)

		if self._items_sum is None or self._items_sq_sum is None:
			self._items_sum = value.clone()
			self._items_sq_sum = value ** 2
			self._counter = 1
		else:
			self._items_sum += value
			self._items_sq_sum += value ** 2
			self._counter += 1

	def is_empty(self) -> bool:
		return self._counter == 0

	def get_current(self) -> Optional[Tensor]:
		return self.get_std()

	def get_std(self) -> Optional[Tensor]:
		if not self.is_empty():
			std = torch.sqrt(self._items_sq_sum / self._counter - (self._items_sum / self._counter) ** 2)
			if self._unbiased:
				std = std * torch.scalar_tensor(self._counter / (self._counter - 1)).sqrt()
			return std
		else:
			return None

	def get_nb_values_added(self) -> int:
		return self._counter


class MinTracker(IncrementalMetric):
	def __init__(self, *args):
		"""
			Keep the minimum of the values stored.
		"""
		super().__init__()
		self._min = None
		self._index_min = -1
		self._counter = 0

		self.add_values(list(args))

	def reset(self):
		self._min = None
		self._index_min = -1
		self._counter = 0

	def add(self, value: Tensor):
		if not isinstance(value, Tensor):
			value = torch.as_tensor(value)

		if self._min is None or self._min > value:
			self._min = value.clone()
			self._index_min = self._counter
		self._counter += 1

	def is_empty(self) -> bool:
		return self._min is None

	def get_current(self) -> Optional[Tensor]:
		return self.get_min()

	def get_min(self) -> Optional[Tensor]:
		return self._min

	def get_index(self) -> int:
		return self._index_min

	def get_nb_values_added(self) -> int:
		return self._counter


class MaxTracker(IncrementalMetric):
	def __init__(self, *args):
		"""
			Keep the maximum of the values stored.
		"""
		super().__init__()
		self._max = None
		self._index_max = -1
		self._counter = 0

		self.add_values(list(args))

	def reset(self):
		self._max = None
		self._index_max = -1
		self._counter = 0

	def add(self, value: Tensor):
		if not isinstance(value, Tensor):
			value = torch.as_tensor(value)

		if self._max is None or self._max < value:
			self._max = value.clone()
			self._index_max = self._counter
		self._counter += 1

	def is_empty(self) -> bool:
		return self._max is None

	def get_current(self) -> Optional[Tensor]:
		return self.get_max()

	def get_max(self) -> Optional[Tensor]:
		return self._max

	def get_index(self) -> int:
		return self._index_max

	def get_nb_values_added(self) -> int:
		return self._counter
