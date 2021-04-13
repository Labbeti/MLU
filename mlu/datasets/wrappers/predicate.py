
from torch.utils.data.dataset import Subset
from typing import Callable
from mlu.datasets.wrappers.base import DatasetWrapper
from mlu.utils.typing_ import SizedDataset


class PredicateSubset(DatasetWrapper):
	def __init__(self, dataset: SizedDataset, predicate: Callable):
		"""
			Create a subset with all item which complains with predicate.

			:param dataset: The dataset to wrap.
			:param predicate: The predicate on item to keep.
		"""
		indices = [i for i in range(len(dataset)) if predicate(dataset[i])]
		super().__init__(dataset)
		self._subset = Subset(dataset, indices)

	def __getitem__(self, idx: int) -> tuple:
		return self._subset[idx]

	def __len__(self):
		return len(self._subset)
