
from mlu.datasets.wrappers.transform import TransformDataset
from torch.utils.data.dataset import Dataset


class OnlyLabelDataset(TransformDataset):
	def __init__(self, dataset: Dataset, label_idx: int = 1):
		"""
			Wrapper of Dataset which remove label from dataset by getting only the batch.

			:param dataset: The dataset to wrap.
			:param label_idx: The index of the label to keep when after calling getitem() method of the dataset wrapped.
		"""
		super().__init__(dataset=dataset, transform=lambda item: item[label_idx])
