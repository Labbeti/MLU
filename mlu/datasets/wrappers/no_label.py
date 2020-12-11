
from torch.utils.data import Dataset
from typing import Any


class NoLabelDataset(Dataset):
	def __init__(self, dataset: Dataset, data_idx: int = 0):
		"""
			Wrapper of Dataset which remove label from dataset by getting only the batch.

			:param dataset: The dataset to wrap.
			:param data_idx: The index of the data to keep when after calling getitem() method of the dataset wrapped.
		"""
		super().__init__()
		self.dataset = dataset
		self.data_idx = data_idx

	def __getitem__(self, idx: Any) -> Any:
		return self.dataset[idx][self.data_idx]

	def __len__(self) -> int:
		return len(self.dataset)
