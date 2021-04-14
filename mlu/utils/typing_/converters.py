
import math

from typing import Callable, Optional, Union


def str_to_bool() -> Callable[[str], bool]:
	"""
		Convert a string to bool. Case insensitive.

		- x in ['true', '1', 'yes', 'y'] => True,
		- x in ['false', '0', 'no', 'n'] => False,
		- _ => RuntimeError

		:returns: The function for convert a value to the corresponding boolean value.
	"""
	def str_to_bool_impl(x: str) -> bool:
		x_low = str(x).lower()
		if x_low in ['true', '1', 'yes', 'y']:
			return True
		elif x_low in ['false', '0', 'no', 'n']:
			return False
		else:
			raise RuntimeError('Invalid boolean argument "{:s}".'.format(x))
	return str_to_bool_impl


def str_to_optional_bool() -> Callable[[str], Optional[bool]]:
	"""
		Convert a string to optional bool value. Case insensitive.

		- x in ['none'] => None
		- x in ['true', '1', 'yes', 'y'] => True
		- x in ['false', '0', 'no', 'n'] => False
		- _ => RuntimeError

		:returns: The function for convert a value to the corresponding boolean or None value.
	"""
	def str_to_optional_bool_impl(x: str) -> Optional[bool]:
		x_low = str(x).lower()
		if x_low in ['none']:
			return None
		else:
			return str_to_bool()(x)
	return str_to_optional_bool_impl


def str_to_optional_str() -> Callable[[str], Optional[str]]:
	"""
		Convert string to optional string value. Case insensitive.

		:returns: The function for convert a value to None if x == 'None', otherwise the string value.
	"""
	def str_to_optional_str_impl(x: str) -> Optional[str]:
		x_low = str(x).lower()
		if x_low in ['none']:
			return None
		else:
			return x
	return str_to_optional_str_impl


def str_to_optional_int() -> Callable[[str], Optional[int]]:
	"""
		Convert string to optional integer value. Case insensitive.

		:returns: The function for convert a value to integer value, None or throw ValueError exception.
	"""
	def str_to_optional_int_impl(x: str) -> Optional[int]:
		x_low = str(x).lower()
		if x_low in ['none']:
			return None
		else:
			return int(x)
	return str_to_optional_int_impl


def str_to_optional_float() -> Callable[[str], Optional[float]]:
	"""
		Convert string to optional float value. Case insensitive.

		:returns: The function for convert a value to float value, None or throw ValueError exception.
	"""
	def str_to_optional_float_impl(x: str) -> Optional[float]:
		x = str(x)
		if x.lower() == 'none':
			return None
		else:
			return float(x)
	return str_to_optional_float_impl


def str_to_union_str_int() -> Callable[[str], Union[str, int]]:
	"""
		Convert string to integer value or string value.

		:returns: The function for convert a value, return the integer value, otherwise returns a the same string value.
	"""
	def str_to_union_str_int_impl(x: str) -> Union[str, int]:
		x = str(x)
		try:
			x_int = int(x)
			return x_int
		except ValueError:
			return x
	return str_to_union_str_int_impl


def float_in_range(
	min_: float,
	max_: float,
	include_min: bool = True,
	include_max: bool = False,
) -> Callable[[str], float]:
	"""
		Convert string to float value and check his range.

		:param min_: The minimal float value.
		:param max_: The maximal float value.
		:param include_min: If true, the minimal value in included. (default: True)
		:param include_max: If true, the maximal value in included. (default: False)
		:returns: The function for checking the float range.
	"""
	def float_in_range_impl(x: str) -> float:
		x = float(x)
		if min_ < x < max_ or (include_min and x == min_) or (include_max and x == max_):
			return x
		else:
			raise ValueError(
				'Value "{:s}" is not a float in range {:s}{:f},{:f}{:s}'.format(
					str(x), '[' if include_min else ']', min_, max_, ']' if include_max else '['
				)
			)
	return float_in_range_impl


def positive_float() -> Callable[[str], float]:
	"""
		Convert string to float value and check the range [0, inf[.

		:returns: The function for checking the float range.
	"""
	return float_in_range(0.0, math.inf, True, False)


def strict_positive_float() -> Callable[[str], float]:
	"""
		Convert string to float value and check the range ]0, inf[.

		:returns: The function for checking the float range.
	"""
	return float_in_range(0.0, math.inf, False, False)
