
from torch.utils.data.sampler import Sampler
from typing import Iterable, List, Optional, Sized


class SubsetSampler(Sampler):
	"""
		A subset sampler without shuffle.
	"""
	def __init__(self, indexes: List[int], data_source: Optional[Sized] = None):
		super().__init__(data_source)
		self._indexes = indexes

	def __iter__(self) -> Iterable:
		return iter(self._indexes)

	def __len__(self) -> int:
		return len(self._indexes)
