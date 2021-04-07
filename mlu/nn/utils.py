
import torch

from torch import Tensor
from torch.nn import Module
from typing import Callable


def identity(x: Tensor) -> Tensor:
	return x


def batchmean(x: Tensor) -> Tensor:
	return torch.mean(x, dim=-1)


def get_reduction_from_name(name: str) -> Callable[[Tensor], Tensor]:
	"""
		:param name: The name of the reduction function.
			Available functions are 'sum' and 'mean', 'none' and 'batchmean'.
		:return: The reduction function with a name.
	"""
	if name in ["mean"]:
		return torch.mean
	elif name in ["sum"]:
		return torch.sum
	elif name in ["none", "identity"]:
		return identity
	elif name in ["batchmean"]:
		return batchmean
	else:
		raise RuntimeError(f"Unknown reduction '{name}'. Must be one of {str(['mean', 'sum', 'none', 'batchmean'])}.")


def get_module_checksum(model: Module, only_trainable: bool = True) -> Tensor:
	params = (param for param in model.parameters() if not only_trainable or param.requires_grad)
	return sum(param.sum() for param in params)


def get_num_parameters(model: Module, only_trainable: bool = True) -> int:
	"""
		Return the number of parameters in a module.

		:param model: Pytorch Module to check.
		:param only_trainable: If True, count only parameters that requires gradient. (default: True)
		:returns: The number of parameters.
	"""
	params = (param for param in model.parameters() if not only_trainable or param.requires_grad)
	return sum(param.numel() for param in params)


def check_module_shapes(m1: Module, m2: Module, only_trainable: bool = True) -> bool:
	params1 = [param for param in m1.parameters() if not only_trainable or param.requires_grad]
	params2 = [param for param in m2.parameters() if not only_trainable or param.requires_grad]

	return len(params1) == len(params2) and all(p1.shape == p2.shape for p1, p2 in zip(params1, params2))


def check_module_equal(m1: Module, m2: Module, only_trainable: bool = True) -> bool:
	params1 = [param for param in m1.parameters() if not only_trainable or param.requires_grad]
	params2 = [param for param in m2.parameters() if not only_trainable or param.requires_grad]

	return len(params1) == len(params2) and all([p1.shape == p2.shape and p1.eq(p2).all() for p1, p2 in zip(params1, params2)])
