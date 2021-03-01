
from torch.optim.optimizer import Optimizer
from typing import List


def get_lrs(optim: Optimizer) -> List[float]:
	"""
		Get the learning rates in all groups of an optimizer.

		:param optim: The optimizer to get.
		:return: The list of learning rates.
	"""
	return [group["lr"] for group in optim.param_groups]


def get_lr(optim: Optimizer, idx: int = 0) -> float:
	"""
		Get the learning rate of the first group of an optimizer.

		:param optim: The optimizer to get.
		:param idx: The group index of the learning rate in the optimizer. (default: 0)
		:return: The learning rate.
	"""
	return get_lrs(optim)[idx]
