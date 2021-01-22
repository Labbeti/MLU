
from mlu.datasets.wrappers.transform import TransformDataset
from torch.utils.data.dataset import Dataset


class NoLabelDataset(TransformDataset):
	def __init__(self, dataset: Dataset, data_idx: int = 0):
		"""
			Wrapper of Dataset which remove label from dataset by getting only the batch.

			:param dataset: The dataset to wrap.
			:param data_idx: The index of the data to keep when after calling getitem() method of the dataset wrapped.
		"""
		super().__init__(dataset, transform=lambda item: item[data_idx])
