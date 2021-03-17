
from torch.utils.data.dataset import Dataset
from typing import Any, Iterable, Union

from mlu.datasets.wrappers.base import DatasetWrapper


class NoLabelDataset(DatasetWrapper):
	def __init__(self, dataset: Dataset, label_idx: Union[int, Iterable[int]] = 1, item_size: int = 2):
		"""
			Wrapper of Dataset which remove label from dataset by getting only the batch.

			:param dataset: The dataset to wrap.
			:param label_idx: The index of the data to keep when after calling getitem() method of the dataset wrapped.
		"""
		super().__init__(dataset)
		if isinstance(label_idx, int):
			label_idx = [label_idx]
		self.other_idx = list(set(range(item_size)).difference(label_idx))
		assert len(self.other_idx) > 0

	def __getitem__(self, idx: Any) -> Any:
		item = super().__getitem__(idx)
		if len(self.other_idx) == 1:
			return item[self.other_idx[0]]
		else:
			return tuple(item[idx] for idx in self.other_idx)
