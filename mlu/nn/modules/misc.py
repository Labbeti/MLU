
import math
import torch

from torch import Tensor
from torch.nn import Module
from typing import Any, Callable, Dict, List, Optional, Union

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
		return f'dim={self.dim}'


class UnSqueeze(Module):
	def __init__(self, dim: int):
		super().__init__()
		self.dim = dim

	def forward(self, x: Tensor) -> Tensor:
		return x.unsqueeze(self.dim)

	def extra_repr(self) -> str:
		return f'dim={self.dim}'


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
		return torch.clamp(x, min=self.min, max=self.max)

	def extra_repr(self) -> str:
		return f'min={self.min}, max={self.max}'


class Identity(Module):
	def forward(self, *args):
		return args


class ForwardList(List[Module], Module):
	def __init__(self, *args, **kwargs):
		list.__init__(self, *args, **kwargs)
		Module.__init__(self)

	def forward(self, *args, **kwargs) -> List[Any]:
		return [module(*args, **kwargs) for module in self]


class ForwardDict(Dict[str, Module], Module):
	def __init__(self, *args: Union[dict, Callable, None], **kwargs):
		"""
			Compute output of each module stored when forward() is called.
			Subclass of Dict[str, Module] and Module.
		"""
		args = [arg for arg in args if arg is not None]
		dict.__init__(self, *args, **kwargs)
		Module.__init__(self)

	def forward(self, *args, **kwargs) -> Dict[str, Any]:
		return {name: module(*args, **kwargs) for name, module in self.items()}

	def __hash__(self) -> int:
		return hash(tuple(sorted(self.items())))


class ForwardDictAffix(ForwardDict):
	"""
			Compute score of each callable object stored when forward() is applied.
			Subclass of Dict[str, Module] and Module.

			Example :

			>>> import torch
			>>> from mlu.metrics import CategoricalAccuracy, FScore
			>>> from mlu.nn import ForwardDictAffix
			>>> input_ = torch.rand(5, 10)
			>>> target = torch.rand(5, 10)
			>>> metric_dict = ForwardDictAffix(acc=CategoricalAccuracy(), f1=FScore())
			>>> metric_dict(input_, target)
			... {'acc': 0.4, 'f1': 0.1}
	"""
	def __init__(self, *args: Union[dict, Callable, None], prefix: str = "", suffix: str = "", **kwargs):
		super().__init__(*args, **kwargs)
		self.prefix = prefix
		self.suffix = suffix

	def forward(self, *args, **kwargs) -> Dict[str, Any]:
		return {self.prefix + name + self.suffix: output for name, output in super().forward(*args, **kwargs).items()}

	def to_dict(self, with_affixes: bool = True) -> Dict[str, Module]:
		if with_affixes:
			return {self.prefix + name + self.suffix: module for name, module in self.items()}
		else:
			return dict(self)
