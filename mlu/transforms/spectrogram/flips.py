
import torch
from torch import Tensor

from mlu.transforms.base import SpectrogramTransform


class Flip(SpectrogramTransform):
	def __init__(self, dim: int, p: float = 1.0):
		super().__init__(p=p)
		self.dim = dim

	def process(self, x: Tensor) -> Tensor:
		return torch.flip(x, (self.dim,))


class VerticalFlip(Flip):
	def __init__(self, p: float = 1.0):
		super().__init__(dim=-2, p=p)


class HorizontalFlip(Flip):
	def __init__(self, p: float = 1.0):
		super().__init__(dim=-1, p=p)
