
from torch.utils.data.dataset import Dataset
from typing import Any

from mlu.datasets.wrappers.base import DatasetWrapper


class NoLabelDataset(DatasetWrapper):
	def __init__(self, dataset: Dataset, index_label: int = 1, keep_tuple: bool = False):
		"""
			Wrapper of Dataset which remove label from dataset by getting only the batch.

			:param dataset: The dataset to wrap.
			:param index_label: The index of the data to keep when after calling getitem() method of the dataset wrapped.
		"""
		super().__init__(dataset)
		self.index_label = index_label
		self.keep_tuple = keep_tuple

	def __getitem__(self, index: Any) -> Any:
		item: tuple = super().__getitem__(index)
		item = tuple(elt for i, elt in enumerate(item) if i != self.index_label)
		if len(item) == 1 and not self.keep_tuple:
			return item[0]
		else:
			return item
