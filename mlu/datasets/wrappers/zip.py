
from torch.utils.data.dataset import Dataset
from typing import List, Sized


class ZipDataset(Dataset, Sized):
	def __init__(self, datasets: List[Dataset]):
		"""
			Zip through a list of Sized datasets of same sizes.

			:param datasets: The list of dataset to read.
		"""
		super().__init__()
		self._datasets = datasets
		self._check_attributes()

	@staticmethod
	def from_list(datasets: List[Dataset]) -> 'ZipDataset':
		return ZipDataset(datasets)

	def _check_attributes(self):
		assert len(self._datasets) > 0, "At least 1 dataset must be given as argument to ZipDataset."

		for dataset in self._datasets:
			if not isinstance(dataset, Sized):
				raise RuntimeError("Dataset Zipped must be Sized.")

		if len(self._datasets) > 0:
			assert all(
				[len(dataset) == len(self._datasets[0]) for dataset in self._datasets[1:]]
			), "Datasets must have the same size"

	def __getitem__(self, idx: int) -> list:
		return [d[idx] for d in self._datasets]

	def __len__(self) -> int:
		return len(self._datasets[0]) if len(self._datasets) > 0 else 0

	def unwrap(self) -> List[Dataset]:
		return self._datasets
