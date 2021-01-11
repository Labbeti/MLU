
from torch.utils.data.dataset import Dataset
from typing import Any, Callable, Optional


class TransformDataset(Dataset):
	def __init__(self, dataset: Dataset, transform: Optional[Callable], index: Optional[int] = 0):
		"""
			Wrap a dataset by applying a post-transform to item get by the method "__getitem__".

			:param dataset: The dataset to wrap.
			:param transform: The callable transform to apply.
			:param index: The index of the element to apply the transform.
				If None, apply the transform to the complete item.
		"""
		super().__init__()
		self.dataset = dataset
		self.transform = transform
		self.index = index

	def __getitem__(self, idx: Any) -> Any:
		item = self.dataset[idx]
		if self.transform is not None:
			if self.index is None:
				item = self.transform(item)
			else:
				item = list(item)
				item[self.index] = self.transform(item[self.index])
				item = tuple(item)
		return item

	def __len__(self) -> int:
		return len(self.dataset)
