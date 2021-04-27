
from mlu.datasets.wrappers.transform import TransformDataset
from mlu.nn import OneHot
from torch.utils.data.dataset import Dataset
from typing import Optional


class OneHotDataset(TransformDataset):
	def __init__(self, dataset: Dataset, n_classes: int, smooth: Optional[float] = None, index: int = 1):
		"""
			Wrap a dataset and convert labels to one_hot encoding.

			:param dataset: The dataset to wrap.
			:param n_classes: The number of classes in the dataset.
			:param smooth: The optional smoothing label parameter.
			:param index: The index of the label in the item returned by the dataset.
		"""
		super().__init__(
			dataset=dataset,
			transform=OneHot(n_classes=n_classes, smooth=smooth),
			index=index,
		)
