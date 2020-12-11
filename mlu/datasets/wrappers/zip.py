
from torch.utils.data import Dataset
from typing import List


class ZipDataset(Dataset):
	def __init__(self, datasets: List[Dataset]):
		"""
			Zip through a list of Sized datasets of same sizes.

			:param datasets: The list of dataset to read.
		"""
		super().__init__()
		self.datasets = datasets
		self._check_attributes()

	def _check_attributes(self):
		len_ = len(self.datasets[0])
		for d in self.datasets[1:]:
			assert len(d) == len_, "Datasets must have the same size"

	def __getitem__(self, idx: int) -> list:
		return [d[idx] for d in self.datasets]

	def __len__(self) -> int:
		return len(self.datasets[0]) if len(self.datasets) > 0 else 0
