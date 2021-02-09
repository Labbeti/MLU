
from mlu.datasets.wrappers.transform import TransformDataset
from torch.utils.data.dataset import Dataset


class NoLabelDataset(TransformDataset):
	def __init__(self, dataset: Dataset, label_idx: int = 1, item_size: int = 2):
		"""
			Wrapper of Dataset which remove label from dataset by getting only the batch.

			:param dataset: The dataset to wrap.
			:param label_idx: The index of the data to keep when after calling getitem() method of the dataset wrapped.
		"""
		other_idx = list(set(range(item_size)).difference({label_idx}))
		super().__init__(dataset, transform=lambda item: item[other_idx])
