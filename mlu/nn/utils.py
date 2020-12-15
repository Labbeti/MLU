
import torch

from torch import Tensor
from typing import Callable


def get_reduction_from_name(name: str) -> Callable[[Tensor], Tensor]:
	"""
		Return the reduction function with a name.

		:param name: The name of the reduction function.
			Available functions are 'sum' and 'mean' and 'none'.
	"""
	if name in ["mean"]:
		return torch.mean
	elif name in ["sum"]:
		return torch.sum
	elif name in ["none"]:
		return lambda x: x
	else:
		raise RuntimeError("Unknown reduction \"{:s}\". Must be in {:s}.".format(name, str(["mean", "sum", "none"])))
