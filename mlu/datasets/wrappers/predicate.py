
from mlu.datasets.wrappers.base import DatasetWrapper
from torch.utils.data.dataset import Dataset, Subset
from typing import Callable, Sized


class PredicateSubset(DatasetWrapper):
	def __init__(self, dataset: Dataset, predicate: Callable):
		"""
			Create a subset with all item which complains with predicate.

			:param dataset: The dataset to wrap.
			:param predicate: The predicate on item to keep.
		"""
		indices = [i for i in range(len(dataset)) if predicate(dataset[i])]
		subset = Subset(dataset, indices)
		super().__init__(subset)
