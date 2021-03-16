
import torch

from torch.utils.data.sampler import Sampler, SubsetRandomSampler
from typing import Iterator, List, Optional, Sized


class SubsetSampler(Sampler):
	"""
		A subset sampler without shuffle.
	"""
	def __init__(self, indexes: List[int], data_source: Optional[Sized] = None):
		super().__init__(data_source)
		self._indexes = indexes

	def __iter__(self) -> Iterator[int]:
		return iter(self._indexes)

	def __len__(self) -> int:
		return len(self._indexes)


class SubsetRandomCycleSampler(SubsetRandomSampler):
	def __init__(
		self,
		indexes: List[int],
		nb_max_iterations: Optional[int] = None,
	):
		super().__init__(indexes)
		self.nb_max_iterations = nb_max_iterations if nb_max_iterations is not None else len(indexes)

	def __iter__(self) -> Iterator[int]:
		global_idx = 0
		finished = False

		while not finished:
			for i in torch.randperm(len(self.indices), generator=self.generator):
				if global_idx >= len(self):
					finished = True
					break

				yield self.indices[i]
				global_idx += 1

	def __len__(self) -> int:
		return self.nb_max_iterations
