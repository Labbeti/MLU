
import torch

from torch import Tensor
from typing import Union


def _add_subname_prefix(name: str, subname_prefix: str) -> str:
	if subname_prefix is not None and subname_prefix != "":
		section, subname = _tag_split(name)
		return f"{section}/{subname_prefix}_{subname}"
	else:
		return name


def _add_subname_suffix(name: str, subname_suffix: str) -> str:
	if subname_suffix is not None and subname_suffix != "":
		return f"{name}_{subname_suffix}"
	else:
		return name


def _add_section_suffix(name: str, section_suffix: str) -> str:
	if section_suffix is not None and section_suffix != "":
		section, subname = _tag_split(name)
		return f"{section}_{section_suffix}/{subname}"
	else:
		return name


def _add_pre_and_suffix(name: str, section_suffix: str, subname_prefix: str) -> str:
	name = _add_section_suffix(name, section_suffix)
	name = _add_subname_prefix(name, subname_prefix)
	return name


def _tag_split(name: str) -> (str, str):
	split = name.split("/")
	if len(split) == 1:
		return "", split[0]
	elif len(split) == 2:
		return split[0], split[1]
	else:
		raise RuntimeError(f"Found more than 2 '/' in recorder tag '{name}'.")


def _to_float(scalar: Union[float, Tensor]) -> float:
	if isinstance(scalar, Tensor):
		if torch.as_tensor(scalar, dtype=torch.int).prod() != 1:
			raise RuntimeError(f"Cannot add a non-scalar tensor of shape {str(scalar.shape)}.")
		return scalar.item()
	else:
		return scalar
