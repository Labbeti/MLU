
from torch.utils.data.dataset import Dataset
from typing import List

from mlu.utils.typing_ import SizedDataset


class ZipDataset(Dataset):
	def __init__(self, *datasets: SizedDataset):
		"""
			Zip through a list of Sized datasets of same sizes.

			:param datasets: The list of dataset to read.
		"""
		assert len(self._datasets) > 0, "At least 1 dataset must be given as argument to ZipDataset."

		super().__init__()
		self._datasets = list(datasets)

	@staticmethod
	def from_list(datasets: List[SizedDataset]) -> 'ZipDataset':
		return ZipDataset(*datasets)

	def __getitem__(self, idx: int) -> list:
		return [d[idx] for d in self._datasets]

	def __len__(self) -> int:
		return min(len(dataset) for dataset in self._datasets)

	def unwrap(self) -> List[SizedDataset]:
		return self._datasets
