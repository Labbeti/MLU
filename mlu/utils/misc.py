
import numpy as np
import random
import subprocess
import torch

from datetime import datetime
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.tensorboard import SummaryWriter
from typing import Any, Dict, List, Sequence, Tuple, Union


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
		:param width_range: The width ratio range of the rectangle.
			Ex: (0.1, 0.5) => width is sampled from (0.1 * width, 0.5 * width).
		:param height_range: The height ratio range of the rectangle.
			Ex: (0.0, 0.9) => height is sampled from (0.0, 0.9 * height).
		:returns: The limits (left, right, top, down) of the rectangle created.
	"""
	assert 0.0 <= width_range[0] <= width_range[1] <= 1.0
	assert 0.0 <= height_range[0] <= height_range[1] <= 1.0

	min_width = max(int(width_range[0] * width_img), 1)
	min_height = max(int(height_range[0] * height_img), 1)

	max_width = max(int(width_range[1] * width_img), min_width + 1)
	max_height = max(int(height_range[1] * height_img), min_height + 1)

	width = torch.randint(low=min_width, high=max_width, size=()).item()
	height = torch.randint(low=min_height, high=max_height, size=()).item()

	max_left = max(width_img - width, 1)
	max_top = max(height_img - height, 1)

	left = torch.randint(low=0, high=max_left, size=()).item()
	top = torch.randint(low=0, high=max_top, size=()).item()
	right = left + width
	down = top + height

	return left, right, top, down


def random_cuboid(shapes: Sequence[int], ratios: Sequence[Tuple[float, float]]) -> List[Tuple[int, int]]:
	"""
		Random cuboid generated using ratios.

		:param shapes: The shape of the cuboid as sequence of ints. Size: (N,).
		:param ratios: The list of min and max ratios for each dim for sampling the cuboid shape. Size: (N, 2)
		:returns: The limits of the cuboid with limits on each dimension. Size: (N, 2)
	"""
	assert all((0.0 <= min_ <= max_ <= 1.0 for min_, max_ in ratios))
	assert len(shapes) == len(ratios)

	limits = []
	for length, (min_, max_) in zip(shapes, ratios):
		min_len = int(min_ * length)
		max_len = max(int(max_ * length), min_len + 1)
		rand_len = torch.randint(low=min_len, high=max_len, size=()).item()

		rand_left_max = max(length - rand_len, 1)
		rand_left = torch.randint(low=0, high=rand_left_max, size=()).item()
		rand_right = rand_left + rand_len
		limits.append((rand_left, rand_right))

	return limits


def get_lrs(optim: Optimizer) -> List[float]:
	""" Get the learning rates of an optimizer. """
	return [group["lr"] for group in optim.param_groups]


def get_lr(optim: Optimizer, idx: int = 0) -> float:
	""" Get the learning rate of an optimizer. """
	return get_lrs(optim)[idx]


def get_nb_parameters(model: Module, trainable_param: bool = True) -> int:
	"""
		Return the number of parameters in a model.

		:param model: Pytorch Module to check.
		:param trainable_param: If True, count only parameter that requires gradient. (default: True)
		:returns: The number of parameters.
	"""
	params = (p for p in model.parameters() if not trainable_param or p.requires_grad)
	return sum([p.numel() for p in params])


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


def get_current_git_hash() -> str:
	"""
		Return the current git hash in the current directory.

		:returns: The git hash. If an error occurs, returns 'UNKNOWN'.
	"""
	try:
		git_hash = subprocess.check_output(["git", "describe", "--always"])
		git_hash = git_hash.decode("UTF-8").replace("\n", "")
		return git_hash
	except subprocess.CalledProcessError:
		return "UNKNOWN"
