
import torch

from torch import Tensor
from typing import Tuple, Union
from mlu.transforms.base import SpectrogramTransform


class Flip(SpectrogramTransform):
	def __init__(self, dim: Union[int, Tuple[int, ...]], p: float = 1.0):
		super().__init__(p=p)
		self.dim = dim

	def process(self, data: Tensor) -> Tensor:
		if isinstance(self.dim, int):
			dims = (self.dim,)
		else:
			dims = tuple(self.dim)
		return torch.flip(data, dims)


class VerticalFlip(Flip):
	def __init__(self, p: float = 1.0):
		super().__init__(dim=-2, p=p)


class HorizontalFlip(Flip):
	def __init__(self, p: float = 1.0):
		super().__init__(dim=-1, p=p)
