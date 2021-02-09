
from torch.utils.data.dataset import Dataset
from typing import Any, Callable, Optional, Sized


class TransformDataset(Dataset):
	def __init__(self, dataset: Dataset, transform: Optional[Callable], index: Optional[int] = None):
		"""
			Wrap a dataset by applying a post-transform to item get by the method "__getitem__".

			:param dataset: The dataset to wrap.
			:param transform: The callable transform to apply.
			:param index: The index of the element to apply the transform.
				If None, apply the transform to the complete item.
		"""
		super().__init__()
		self._dataset = dataset
		self._transform = transform
		self._index = index

		if self._transform is None:
			self._post_fn = lambda x: x
		elif self._index is None:
			self._post_fn = self._transform
		else:
			def post_fn(item: tuple) -> tuple:
				item = list(item)
				item[self._index] = self._transform(item[self._index])
				item = tuple(item)
				return item
			self._post_fn = post_fn

	def __getitem__(self, idx: Any) -> Any:
		return self._post_fn(self._dataset[idx])

	def __len__(self) -> int:
		if not isinstance(self._dataset, Sized):
			raise RuntimeError("Wrapped dataset is not Sized (i.e. does not have the method '__len__').")
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
			while isinstance(dataset, TransformDataset):
				dataset = dataset.unwrap(recursive=False)
			return dataset
