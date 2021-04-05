
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


def model_checksum(model: Module) -> Tensor:
	with torch.no_grad():
		parameters = [param.sum() for param in model.parameters() if param.requires_grad]
		return torch.stack(parameters).sum()


def get_num_parameters(model: Module, only_trainable: bool = True) -> int:
	"""
		Return the number of parameters in a model.

		:param model: Pytorch Module to check.
		:param only_trainable: If True, count only parameter that requires gradient. (default: True)
		:returns: The number of parameters.
	"""
	params = (p for p in model.parameters() if not only_trainable or p.requires_grad)
	return sum(p.numel() for p in params)


def check_params_shapes(m1: Module, m2: Module, only_trainable: bool = True) -> bool:
	params1 = [p for p in m1.parameters() if not only_trainable or p.requires_grad]
	params2 = [p for p in m2.parameters() if not only_trainable or p.requires_grad]

	if len(params1) != len(params2):
		return False
	else:
		return all(p1.shape == p2.shape for p1, p2 in zip(params1, params2))
