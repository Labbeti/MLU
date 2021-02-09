
from mlu.datasets.wrappers.base import DatasetWrapper
from torch.utils.data.dataset import Dataset, Subset
from typing import Callable


class PredicateSubset(DatasetWrapper):
	def __init__(self, dataset: Dataset, predicate: Callable):
		"""
			Create a subset with all item which complains with predicate.

			:param dataset: The dataset to wrap.
			:param predicate: The predicate on item to keep.
		"""
		indices = [i for i, item in enumerate(dataset) if predicate(item)]
		subset = Subset(dataset, indices)
		super().__init__(subset)
