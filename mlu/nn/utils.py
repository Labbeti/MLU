
import torch

from torch import Tensor
from typing import Callable


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
	elif name in ["none"]:
		return lambda x: x
	elif name in ["batchmean"]:
		return lambda x: torch.mean(x, dim=-1)
	else:
		raise RuntimeError(f"Unknown reduction '{name}'. Must be one of {str(['mean', 'sum', 'none', 'batchmean'])}.")
