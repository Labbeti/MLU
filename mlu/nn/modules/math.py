
import numpy as np
import torch

from mlu.nn.functional.labels import nums_to_smooth_onehot
from mlu.nn.functional.math import mish

from torch import Tensor
from torch.nn import Module
from typing import Any, Callable, Optional, Union


DEFAULT_EPSILON = 2e-20


class OneHot(Module):
	def __init__(self, nb_classes: int, smooth: Optional[float] = 0.0):
		"""
			Convert label to one-hot encoding.

			:param nb_classes: The number of classes in the dataset.
			:param smooth: The optional label smoothing coefficient parameter.
		"""
		super().__init__()
		self.nb_classes = nb_classes
		self.smooth = smooth if smooth is not None else 0.0

	def forward(self, x: Union[np.ndarray, Tensor]) -> Union[np.ndarray, Tensor]:
		return nums_to_smooth_onehot(x, self.nb_classes, self.smooth)


class Thresholding(Module):
	def __init__(self, threshold: Optional[float], bin_func: Callable = torch.ge):
		"""
			Convert label to multi-hot encoding.

			:param threshold: The threshold used to binarize the input. If None, the forward will have no effect.
			:param bin_func: The comparison function used to binarize the Tensor. (default: torch.ge)
		"""
		super().__init__()
		self.threshold = threshold
		self.bin_func = bin_func

	def forward(self, x: Tensor) -> Tensor:
		if self.threshold is not None:
			return self.bin_func(x, self.threshold).to(x.dtype)
		else:
			return x


class Squeeze(Module):
	def __init__(self, dim: Optional[int] = None):
		super().__init__()
		self.dim = dim

	def forward(self, x: Tensor) -> Tensor:
		return x.squeeze(self.dim)


class UnSqueeze(Module):
	def __init__(self, dim: int):
		super().__init__()
		self.dim = dim

	def forward(self, x: Tensor) -> Tensor:
		return x.unsqueeze(self.dim)


class Mish(Module):
	"""
		Mish class for apply mish function.
	"""
	def forward(self, x: Tensor) -> Tensor:
		return mish(x)


class Min(Module):
	def __init__(self, dim: Optional[int] = 1):
		"""
			Minimum module.

			:param dim: Optional dimension. (default: 1)
		"""
		super().__init__()
		self.dim = dim

	def forward(self, x: Tensor, dim: Optional[int] = None) -> Tensor:
		if dim is None:
			dim = self.dim
		return x.min(dim=dim)[0]


class Max(Module):
	def __init__(self, dim: Optional[int] = 1):
		"""
			Maximum module.

			:param dim: Optional dimension. (default: 1)
		"""
		super().__init__()
		self.dim = dim

	def forward(self, x: Tensor, dim: Optional[int] = None) -> Tensor:
		if dim is None:
			dim = self.dim
		return x.max(dim=dim)[0]


class Mean(Module):
	def __init__(self, dim: Optional[int] = 1):
		"""
			Mean module.

			:param dim: Optional dimension. (default: 1)
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
