
from torch.utils.data import Dataset
from typing import Any, Callable, Optional


class TransformDataset(Dataset):
	def __init__(self, dataset: Dataset, transform: Callable, index: Optional[int] = 0):
		super().__init__()
		self.dataset = dataset
		self.transform = transform
		self.index = index

	def __getitem__(self, idx: Any) -> Any:
		item = self.dataset[idx]
		if self.index is None:
			item = self.transform(item)
		else:
			item = list(item)
			item[self.index] = self.transform(item[self.index])
			item = tuple(item)
		return item

	def __len__(self) -> int:
		return len(self.dataset)
