
import itertools
import random

from torch.utils.data.sampler import Sampler
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


class SubsetCycleSampler(Sampler):
	def __init__(self, indexes: List[int], nb_max_iterations: Optional[int] = None, shuffle: bool = True):
		"""
			SubsetRandomSampler that cycle on indexes until a number max of iterations is reached.

			:param indexes: The list of indexes of the items.
			:param nb_max_iterations: The maximal number of iterations. If None, it will be set to the length of indexes
				and the sampler will have the same behaviour than a SubsetRandomSampler.
				(default: None)
			:param shuffle: If True, shuffle the indexes at every len(indexes).
				(default: True)
		"""
		super().__init__(None)
		self.indexes = indexes
		self.nb_max_iterations = nb_max_iterations if nb_max_iterations is not None else len(indexes)
		self.shuffle = shuffle
		self._shuffle()

	def __iter__(self) -> Iterator[int]:
		for i, idx in enumerate(itertools.cycle(self.indexes)):
			if i % len(self.indexes) == len(self.indexes) - 1:
				self._shuffle()

			if i >= self.nb_max_iterations:
				break

			yield idx

	def __len__(self) -> int:
		return self.nb_max_iterations

	def _shuffle(self):
		if self.shuffle:
			random.shuffle(self.indexes)


class SubsetInfiniteCycleSampler(Sampler):
	"""
		SubsetRandomSampler that cycle indefinitely on indexes.
	"""
	def __init__(self, indexes: List[int], shuffle: bool = True):
		super().__init__(None)
		self.indexes = indexes
		self.shuffle = shuffle
		self._shuffle()

	def __iter__(self) -> Iterator[int]:
		for i, idx in enumerate(itertools.cycle(self.indexes)):
			if i % len(self.indexes) == len(self.indexes) - 1:
				self._shuffle()

			yield idx

	def __len__(self) -> int:
		raise NotImplemented("Infinite sampler does not have __len__() method.")

	def _shuffle(self):
		if self.shuffle:
			random.shuffle(self.indexes)
