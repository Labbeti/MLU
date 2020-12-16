
import numpy as np
import random
import torch

from datetime import datetime
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.tensorboard import SummaryWriter
from typing import Any, Dict, List, Tuple, Union


def get_datetime() -> str:
	"""
		Returns the date in a specific format : "YYYY_MM_DD_hh:mm:ss".
		:returns: The current date.
	"""
	now = str(datetime.now())
	return now[:10] + "_" + now[11:-7]


def reset_seed(seed: int):
	"""
		Reset the seed of following packages : random, numpy, torch, torch.cuda, torch.backends.cudnn.
		:param seed: The seed to set.
	"""
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False


def random_rect(
	width_img: int, height_img: int, width_range: Tuple[float, float], height_range: Tuple[float, float]
) -> (int, int, int, int):
	"""
		Create a random rectangle inside an area defined by the limits (left, right, top, down).

		:param width_img: The maximal width.
		:param height_img: The maximal height.
		:param width_range: The width ratio range of the rectangle. Ex: (0.1, 0.5) => width is sampled from (0.1 * width, 0.5 * width).
		:param height_range: The height ratio range of the rectangle. Ex: (0.0, 0.9) => height is sampled from (0.0, 0.9 * height).
		:returns: The limits (left, right, top, down) of the rectangle created.
	"""
	assert 0.0 <= width_range[0] <= width_range[1] <= 1.0
	assert 0.0 <= height_range[0] <= height_range[1] <= 1.0

	width_min, width_max = max(int(width_range[0] * width_img), 1), max(int(width_range[1] * width_img), 2)
	height_min, height_max = max(int(height_range[0] * height_img), 1), max(int(height_range[1] * height_img), 2)

	if width_min != width_max:
		width = torch.randint(low=width_min, high=width_max, size=[1]).item()
	else:
		width = width_min

	if height_min != height_max:
		height = torch.randint(low=height_min, high=height_max, size=[1]).item()
	else:
		height = height_min

	left = torch.randint(low=0, high=width_img - width, size=[1]).item()
	top = torch.randint(low=0, high=height_img - height, size=[1]).item()
	right = left + width
	down = top + height

	return left, right, top, down


def get_lrs(optim: Optimizer) -> List[float]:
	""" Get the learning rates of an optimizer. """
	return [group["lr"] for group in optim.param_groups]


def get_lr(optim: Optimizer, idx: int = 0) -> float:
	""" Get the learning rate of an optimizer. """
	return get_lrs(optim)[idx]


def get_nb_parameters(model: Module) -> int:
	"""
		Return the number of parameters in a model.

		:param model: Pytorch Module to check.
		:returns: The number of parameters.
	"""
	return sum(p.numel() for p in model.parameters())


def get_nb_trainable_parameters(model: Module) -> int:
	"""
		Return the number of trainable parameters in a model.

		:param model: Pytorch Module.
		:returns: The number of trainable parameters.
	"""
	return sum(p.numel() for p in model.parameters() if p.requires_grad)


def add_dict_to_writer(dic: Dict[str, Any], writer: SummaryWriter):
	"""
		Add dictionary content to tensorboard hyperparameters.
	"""
	def filter_(v: Any) -> Union[str, int, float, Tensor]:
		if any([isinstance(v, type_) for type_ in [str, int, float, Tensor]]):
			return v
		else:
			return str(v)
	dic = {k: filter_(v) for k, v in dic.items()}
	writer.add_hparams(hparam_dict=dic, metric_dict={})
