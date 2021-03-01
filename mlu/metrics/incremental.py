
import torch

from mlu.metrics.base import IncrementalMetric

from torch import Tensor
from typing import Callable, List, Optional


class IncrementalMean(IncrementalMetric):
	def __init__(self, *args: Tensor):
		"""
			Compute the continue average of a values.
		"""
		super().__init__()
		self._sum = None
		self._counter = 0

		self.add_values(list(args))

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

	def set_mean(self, mean: Tensor, counter: int):
		self._sum = mean * counter
		self._counter = counter


class IncrementalStd(IncrementalMetric):
	def __init__(self, *args: Tensor, unbiased: bool = False):
		"""
			Compute the continue unbiased Standard Deviation (std).

			:param unbiased: If True, apply the bessel correction to std (like in the default std of pytorch).
				Otherwise return the classic std (like the default std of numpy).
				(default: False)
		"""
		super().__init__()
		self._unbiased = unbiased
		self._items_sum = None
		self._items_sq_sum = None
		self._counter = 0

		self.add_values(list(args))

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
	def __init__(self, *args: Tensor, start_min: Optional[Tensor] = None, start_index_min: int = -1):
		"""
			Keep the minimum of the values stored.

			:param args: The optional arguments to add to MaxTracker.
			:param start_min: The minimum value stored. (default: None)
			:param start_index_min: The index of the min value stored. (default: -1)
		"""
		super().__init__()
		self._min = start_min
		self._index_min = start_index_min
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

	def set_min(self, min_: Tensor):
		self._min = min_

	def set_index(self, index_min: int):
		self._index_min = index_min


class MaxTracker(IncrementalMetric):
	def __init__(self, *args: Tensor, start_max: Optional[Tensor] = None, start_index_max: int = -1):
		"""
			Keep the maximum of the values stored.

			:param args: The optional arguments to add to MaxTracker.
			:param start_max: The maximum value stored. (default: None)
			:param start_index_max: The index of the max value stored. (default: -1)
		"""
		super().__init__()
		self._max = start_max
		self._index_max = start_index_max
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

	def set_max(self, max_: Tensor):
		self._max = max_

	def set_index(self, index_max: int):
		self._index_max = index_max


class BestTracker(IncrementalMetric):
	def __init__(
		self,
		*args: Tensor,
		is_better: Callable[[Tensor, Tensor], bool] = lambda x, y: x > y,
	):
		"""
			Keep the best of the values stored.
		"""
		super().__init__()
		self._is_better = is_better

		self._best = None
		self._index_best = -1
		self._counter = 0

		self.add_values(list(args))

	def reset(self):
		self._best = None
		self._index_best = -1
		self._counter = 0

	def add(self, value: Tensor):
		if not isinstance(value, Tensor):
			value = torch.as_tensor(value)

		if self._best is None or self._is_better(value, self._best):
			self._best = value.clone()
			self._index_best = self._counter
		self._counter += 1

	def is_empty(self) -> bool:
		return self._best is None

	def get_current(self) -> Optional[Tensor]:
		return self._best

	def get_index(self) -> int:
		return self._index_best

	def get_nb_values_added(self) -> int:
		return self._counter


class NBestsTracker(IncrementalMetric):
	def __init__(
		self,
		*args: Tensor,
		is_better: Callable[[Tensor, Tensor], bool] = lambda x, y: x > y,
		n: int = 1,
	):
		super().__init__()
		self._is_better = is_better
		self._n = n

		self._max_list = []
		self._index_max_list = []
		self._counter = 0

		self.add_values(list(args))

	def reset(self):
		self._max_list = []
		self._index_max_list = []
		self._counter = 0

	def add(self, value: Tensor):
		if not isinstance(value, Tensor):
			value = torch.as_tensor(value)

		insert_index = len(self._max_list)
		for i, max_value in enumerate(self._max_list):
			if self._is_better(value, max_value):
				insert_index = i
				break

		self._max_list.insert(insert_index, value)
		self._index_max_list.insert(insert_index, self._counter)

		while len(self._max_list) > self._n:
			self._max_list.pop()
			self._index_max_list.pop()

		self._counter += 1

	def is_empty(self) -> bool:
		return len(self._max_list) == 0

	def get_current(self) -> Optional[Tensor]:
		return self.get_max()

	def get_max(self) -> Optional[Tensor]:
		if self.is_empty():
			return None
		else:
			return torch.as_tensor(self._max_list)

	def get_index(self) -> List[int]:
		return self._index_max_list

	def get_nb_values_added(self) -> int:
		return self._counter
