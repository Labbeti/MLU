
import torch

from torch import Tensor
from typing import Union


def tag_split(name: str) -> (str, str):
	split = name.split("/")
	if len(split) == 1:
		return "", split[0]
	else:
		return split[0], "/".join(split[1:])


def _add_section_suffix(name: str, section_suffix: str) -> str:
	if section_suffix is not None and section_suffix != "":
		section, subname = tag_split(name)
		return f"{section}_{section_suffix}/{subname}"
	else:
		return name


def _add_subname_suffix(name: str, subname_suffix: str) -> str:
	if subname_suffix is not None and subname_suffix != "":
		return f"{name}/{subname_suffix}"
	else:
		return name


def _add_suffixes(name: str, section_suffix: str, subname_prefix: str) -> str:
	name = _add_section_suffix(name, section_suffix)
	name = _add_subname_suffix(name, subname_prefix)
	return name


def _to_float(scalar: Union[float, Tensor]) -> float:
	if isinstance(scalar, Tensor):
		if torch.as_tensor(scalar, dtype=torch.int).prod() != 1:
			raise RuntimeError(f"Cannot add a non-scalar tensor of shape {str(scalar.shape)}.")
		return scalar.item()
	else:
		return scalar
