
import math
import torch

from torch import Tensor
from torch.nn import Module
from typing import Any, Optional

from mlu.nn.functional.misc import mish


DEFAULT_EPSILON = 2e-20


class Squeeze(Module):
	def __init__(self, dim: Optional[int] = None):
		super().__init__()
		self.dim = dim

	def forward(self, x: Tensor) -> Tensor:
		return x.squeeze(self.dim)

	def extra_repr(self) -> str:
		return f"dim={self.dim}"


class UnSqueeze(Module):
	def __init__(self, dim: int):
		super().__init__()
		self.dim = dim

	def forward(self, x: Tensor) -> Tensor:
		return x.unsqueeze(self.dim)

	def extra_repr(self) -> str:
		return f"dim={self.dim}"


class Mish(Module):
	"""
		Mish class for apply mish function.
	"""
	def forward(self, x: Tensor) -> Tensor:
		return mish(x)


class Min(Module):
	def __init__(self, dim: Optional[int] = None):
		"""
			Minimum module.

			:param dim: Optional dimension. (default: None)
		"""
		super().__init__()
		self.dim = dim

	def forward(self, x: Tensor, dim: Optional[int] = None) -> Tensor:
		if dim is None:
			dim = self.dim
		return x.min(dim=dim)[0]


class Max(Module):
	def __init__(self, dim: Optional[int] = None):
		"""
			Maximum module.

			:param dim: Optional dimension. (default: None)
		"""
		super().__init__()
		self.dim = dim

	def forward(self, x: Tensor, dim: Optional[int] = None) -> Tensor:
		if dim is None:
			dim = self.dim
		return x.max(dim=dim)[0]


class Mean(Module):
	def __init__(self, dim: Optional[int] = None):
		"""
			Mean module.

			:param dim: Optional dimension. (default: None)
		"""
		super().__init__()
		self.dim = dim

	def forward(self, x: Tensor, dim: Optional[int] = None) -> Tensor:
		if dim is None:
			dim = self.dim
		return x.mean(dim=dim)


class Permute(Module):
	def __init__(self, *dims: int):
		super().__init__()
		self.dims = list(dims)

	def forward(self, x: Tensor) -> Tensor:
		out = x.permute(*self.dims)
		return out


class To(Module):
	def __init__(self, *args: Any):
		super().__init__()
		self.args = list(args)

	def forward(self, x: Tensor) -> Tensor:
		return x.to(*self.args)


class Item(Module):
	def forward(self, x: Tensor) -> float:
		return x.item()


class Clamp(Module):
	def __init__(self, min_: float = -math.inf, max_: float = math.inf):
		super().__init__()
		self.min = min_
		self.max = max_

	def forward(self, x: Tensor) -> Tensor:
		return torch.clamp(x, self.min, self.max)


class Identity(Module):
	def forward(self, *args):
		return args
