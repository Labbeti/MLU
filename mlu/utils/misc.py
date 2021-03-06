"""
	Miscellaneous functions utilities.
"""

import inspect
import numpy as np
import random
import re
import subprocess
import torch

from datetime import datetime
from IPython.display import Audio, display
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from types import MethodType, FunctionType, ModuleType
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, TypeVar, Union

T = TypeVar('T')


def get_datetime() -> str:
	"""
		Returns the date in a specific format : 'YYYY_MM_DD_hh:mm:ss'.

		:returns: The current date.
	"""
	now = str(datetime.now())
	return now[:10] + '_' + now[11:-7]


def reset_seed(seed: Optional[int]):
	"""
		Reset the seed of following packages :
			- random
			- numpy
			- torch
			- torch.cuda

		Also set deterministic behaviour for cudnn backend.

		:param seed: The seed to set.
	"""
	if seed is not None:
		random.seed(seed)
		np.random.seed(seed)
		torch.manual_seed(seed)
		torch.cuda.manual_seed_all(seed)

		if hasattr(torch.backends, 'cudnn'):
			torch.backends.cudnn.deterministic = True
			torch.backends.cudnn.benchmark = False
		else:
			raise RuntimeError(
				'Cannot make deterministic behaviour for current torch backend (torch.backends does have the attribute "cudnn").'
			)


def random_rect(
		width_img: int,
		height_img: int,
		width_range: Tuple[float, float],
		height_range: Tuple[float, float]
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
		git_hash = subprocess.check_output(['git', 'describe', '--always'])
		git_hash = git_hash.decode('UTF-8').replace('\n', "")
		return git_hash
	except subprocess.CalledProcessError:
		return 'UNKNOWN'


def to_dict_rec(obj: Any, class_name_key: Optional[str] = '__class__') -> Union[dict, list]:
	"""
		Convert an object to a dictionary.

		Source code was imported from : (with few changes)
			https://stackoverflow.com/questions/1036409/recursively-convert-python-object-graph-to-dictionary

		:param obj: The object to convert.
		:param class_name_key: Key used to save the class name if we convert an object.
		:returns: The dictionary corresponding to the object.
	"""
	if isinstance(obj, dict):
		return {
			key: to_dict_rec(value, class_name_key)
			for key, value in obj.items()
		}
	elif isinstance(obj, Tensor):
		return to_dict_rec(obj.tolist(), class_name_key)
	elif hasattr(obj, '_ast'):
		return to_dict_rec(obj._ast(), class_name_key)
	elif hasattr(obj, '__iter__') and not isinstance(obj, str):
		return [to_dict_rec(v, class_name_key) for v in obj]
	elif hasattr(obj, '__dict__'):
		data = {}
		if class_name_key is not None and hasattr(obj, '__class__'):
			data[class_name_key] = obj.__class__.__name__
		data.update({
			attr: to_dict_rec(value, class_name_key)
			for attr, value in obj.__dict__.items()
			if not callable(value) and not attr.startswith('__')
		})
		return data
	else:
		return obj


def scalar_interpolation(min_: T, max_: T, coefficient: T) -> T:
	"""
		Compute the linear interpolation between min_ and max_ with a coefficient.

		:param min_: The minimal value used for interpolation.
		:param max_: The maximal value used for interpolation.
		:param coefficient: The coefficient in [0.0, 1.0] for compute the results.
		:returns: The value interpolated between min_ and max_.
	"""
	return (max_ - min_) * coefficient + min_


def scalar_normalization(value: T, old_min: T, old_max: T, new_min: T = 0.0, new_max: T = 1.0) -> T:
	"""
		Normalize a value from range [old_min, old_max] to [new_min, new_max].

		:param value: The value to normalize.
		:param old_min: The minimal value of the previous range.
		:param old_max: The maximal value of the previous range.
		:param new_min: The minimal value of the new range. (default: 0.0)
		:param new_max: The maximal value of the new range. (default: 1.0)
		:returns: The value normalized in the new range.
	"""
	return (value - old_min) / (old_max - old_min) * (new_max - new_min) + new_min


def duration_formatter(seconds: int, format_: str = '%jd:%Hh:%Mm:%Ss') -> str:
	rest = seconds

	rest, seconds = divmod(rest, 60)
	rest, minutes = divmod(rest, 60)
	rest, hours = divmod(rest, 24)
	days = rest

	replaces = {
		'%S': seconds,
		'%M': minutes,
		'%H': hours,
		'%j': days,
	}
	result = format_
	for directive, value in replaces.items():
		result = result.replace(directive, str(value))
	return result


def duration_unformatter(string: str, format_: str = '%jd:%Hh:%Mm:%Ss') -> int:
	replaces = {
		'%S': '(?P<S>[0-9]+)',
		'%M': '(?P<M>[0-9]+)',
		'%H': '(?P<H>[0-9]+)',
		'%j': '(?P<j>[0-9]+)',
	}
	format_re = format_
	for directive, value in replaces.items():
		format_re = format_re.replace(directive, str(value))

	match = re.search(format_re, string)
	if match is None:
		raise RuntimeError(f'Invalid string "{string}" with format "{format_}".')

	seconds = int(match['S'])
	minutes = int(match['M'])
	hours = int(match['H'])
	days = int(match['j'])
	total_seconds = seconds + minutes * 60 + hours * 3600 + days * 3600 * 24

	return total_seconds


def get_func_params_names(func: Union[MethodType, FunctionType, Callable]) -> List[str]:
	"""
		:returns: The name of the parameters of a function.
	"""
	parameters_names = func.__code__.co_varnames
	return list(parameters_names)


def get_param_names(class_or_func: Callable) -> List[str]:
	if inspect.isfunction(class_or_func):
		func = class_or_func
	elif inspect.isclass(class_or_func) and hasattr(class_or_func, '__init__'):
		func = class_or_func.__init__
	elif callable(class_or_func) and hasattr(class_or_func, '__call__'):
		func = class_or_func.__call__
	else:
		raise RuntimeError(
			f'Invalid class, function or object "{class_or_func.__name__}". Must be a function, class or callable object.')

	return get_func_params_names(func)


def filter_dict_with_func(dic: Dict[str, Any], func: Callable) -> Dict[str, Any]:
	param_names = get_param_names(func)
	names_intersection = set(param_names).intersection(dic.keys())
	return {name: dic[name] for name in names_intersection}


def collate_dict_item(items: List[Dict[str, Any]]) -> Dict[str, List[Any]]:
	return {
		key: [item[key] for item in items]
		for key in items[0].keys()
	}


def search_function_in_module(func_name: str, module: ModuleType) -> Optional[Callable]:
	if not inspect.ismodule(module):
		raise RuntimeError(f'Object "{module.__name__}" is not a Module.')

	predicate = lambda member: (
			inspect.isfunction(member) and member.__module__ == module.__name__
	)
	functions = inspect.getmembers(module, predicate)
	functions = [func for name, func in functions if name == func_name]

	if len(functions) == 0:
		return None
	elif len(functions) == 1:
		assert inspect.isfunction(functions[0])
		return functions[0]
	else:
		raise RuntimeError(f'Found multiple functions matching the following name : "{func_name}".')


def play_audio(waveform, sample_rate):
	waveform = waveform.numpy()

	num_channels, num_frames = waveform.shape
	if num_channels == 1:
		display((Audio(waveform[0], rate=sample_rate),))
	elif num_channels == 2:
		display((Audio((waveform[0], waveform[1]), rate=sample_rate),))
	else:
		raise ValueError("Waveform with more than 2 channels are not supported.")
