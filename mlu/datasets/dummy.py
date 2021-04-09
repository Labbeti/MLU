
import torch

from torch import Tensor
from torch.utils.data.dataset import Dataset
from typing import Optional, Sequence, Tuple


class DummyDataset(Dataset):
	def __init__(
		self,
		length: int = 100,
		data_shape: Optional[Sequence[int]] = None,
		num_classes: int = 10,
		balanced: bool = True,
	):
		super().__init__()
		self.length = length
		self.data_shape = data_shape if data_shape is not None else [3, 4]
		self.num_classes = num_classes
		self.balanced = balanced

		if balanced:
			self.targets = torch.as_tensor([i % self.num_classes for i in range(length)])
		else:
			self.targets = torch.randint(low=0, high=self.num_classes, size=(length,))

	def __getitem__(self, idx: int) -> Tuple[Tensor, int]:
		data = self.get_data(idx)
		label = self.get_target(idx)
		return data, label

	def __len__(self) -> int:
		return self.length

	def get_data(self, idx: int) -> Tensor:
		return torch.full(self.data_shape, fill_value=idx)

	def get_target(self, idx: int) -> int:
		return self.targets[idx].item()
