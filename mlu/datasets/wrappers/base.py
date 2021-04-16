
from torch.utils.data.dataset import Dataset
from typing import Any, Sized


class DatasetWrapper(Dataset, Sized):
	def __init__(self, dataset: Dataset):
		super().__init__()
		self._dataset = dataset

	def __getitem__(self, idx: Any) -> Any:
		return self._dataset.__getitem__(idx)

	def __len__(self) -> int:
		if not isinstance(self._dataset, Sized):
			raise NotImplementedError(
				f'Wrapped dataset "{str(type(self._dataset))}" is not Sized (it does not have the method "__len__").'
			)
		return len(self._dataset)

	def unwrap(self, recursive: bool = False) -> Dataset:
		"""
			:param recursive: If True and the dataset wrapped is another TransformDataset, unwrap until the wrapped
				element is not a TransformDataset. (default: False)
		"""
		if not recursive:
			return self._dataset
		else:
			dataset = self._dataset
			while isinstance(dataset, DatasetWrapper):
				dataset = dataset.unwrap()
			return dataset
