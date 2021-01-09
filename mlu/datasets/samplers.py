
from torch.utils.data.sampler import Sampler
from typing import Iterable, List, Sized


class SubsetSampler(Sampler):
	def __init__(self, data_source: Sized, indexes: List[int]):
		super().__init__(data_source)
		self._indexes = indexes

	def __iter__(self) -> Iterable:
		return iter(self._indexes)

	def __len__(self) -> int:
		return len(self._indexes)
