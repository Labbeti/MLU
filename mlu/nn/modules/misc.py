
import math
import torch

from torch import Tensor
from torch.nn import Module
from typing import Any, Callable, Dict, Optional, Union

from mlu.nn.functional.misc import mish


DEFAULT_EPSILON = 2e-20


class Squeeze(Module):
	def __init__(self, dim: Optional[int] = None):
		super().__init__()
		self.dim = dim

	def forward(self, x: Tensor) -> Tensor:
		if self.dim is None:
			return torch.squeeze(x)
		else:
			return torch.squeeze(x, self.dim)

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

	def forward(self, x: Tensor) -> Tensor:
		return x.mean(dim=self.dim)


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


class ModuleDict(Dict[str, Module], Module):
	def __init__(self, *args: Union[dict, Callable, None], prefix: str = "", suffix: str = "", **kwargs):
		"""
			Compute output of each module stored when forward() is called.
			Subclass of Dict[str, Module] and Module.
		"""
		args = [arg for arg in args if arg is not None]
		dict.__init__(self, *args, **kwargs)
		Module.__init__(self)
		self.prefix = prefix
		self.suffix = suffix

	def forward(self, *args, **kwargs) -> dict:
		return {
			(self.prefix + metric_name + self.suffix): metric(*args, **kwargs)
			for metric_name, metric in self.items()
		}

	def __hash__(self) -> int:
		return hash(tuple(sorted(self.items()))) + hash(self.prefix) + hash(self.suffix)

	def to_dict(self, with_pre_and_suf: bool = True) -> Dict[str, Module]:
		if with_pre_and_suf:
			dic = {f"{self.prefix}{metric_name}{self.suffix}": metric for metric_name, metric in self.items()}
		else:
			dic = dict(self)
		return dic
