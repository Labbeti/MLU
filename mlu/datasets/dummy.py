
import torch

from torch import Tensor
from torch.utils.data.dataset import Dataset
from typing import List, Optional, Sized, Tuple


class DummyDataset(Dataset, Sized):
	def __init__(self, len_: int = 100, data_shape: Optional[List[int]] = None, num_classes: int = 10):
		super().__init__()
		self.len_ = len_
		self.data_shape = data_shape if data_shape is not None else [3, 4]
		self.num_classes = num_classes

	def __getitem__(self, idx: int) -> Tuple[Tensor, int]:
		return torch.full(self.data_shape, fill_value=idx), idx % self.num_classes

	def __len__(self) -> int:
		return self.len_
