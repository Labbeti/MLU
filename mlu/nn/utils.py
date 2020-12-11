
import torch

from torch import Tensor
from typing import Callable


def get_reduction_from_name(name: str) -> Callable[[Tensor], Tensor]:
	"""
		Return the reduction function with a name. Available function are 'sum' and 'mean'.
	"""
	if name in ["mean", "batchmean"]:
		return torch.mean
	elif name in ["sum"]:
		return torch.sum
	elif name in ["none"]:
		return lambda x: x
	else:
		raise RuntimeError("Unknown reduction \"{:s}\". Must be in {:s}.".format(name, str(["mean", "sum", "none"])))
