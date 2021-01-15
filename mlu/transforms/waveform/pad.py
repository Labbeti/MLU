
import torch

from mlu.transforms.base import WaveformTransform

from torch import Tensor


class PadAlignLeft(WaveformTransform):
	def __init__(self, target_length: int, fill_value: float = 0.0, p: float = 1.0):
		super().__init__(p)
		self.target_length = target_length
		self.fill_value = fill_value

	def apply(self, waveform: Tensor) -> Tensor:
		waveform = pad_align_left(waveform, self.target_length, self.fill_value)
		return waveform


class PadAlignRight(WaveformTransform):
	def __init__(self, target_length: int, fill_value: float = 0.0, p: float = 1.0):
		super().__init__(p)
		self.target_length = target_length
		self.fill_value = fill_value

	def apply(self, waveform: Tensor) -> Tensor:
		waveform = pad_align_right(waveform, self.target_length, self.fill_value)
		return waveform


class PadAlignCenter(WaveformTransform):
	def __init__(self, target_length: int, fill_value: float = 0.0, p: float = 1.0):
		super().__init__(p)
		self.target_length = target_length
		self.fill_value = fill_value

	def apply(self, waveform: Tensor) -> Tensor:
		waveform = pad_align_center(waveform, self.target_length, self.fill_value)
		return waveform


class PadAlignRandom(WaveformTransform):
	def __init__(self, target_length: int, fill_value: float = 0.0, p: float = 1.0):
		super().__init__(p)
		self.target_length = target_length
		self.fill_value = fill_value

	def apply(self, waveform: Tensor) -> Tensor:
		waveform = pad_align_random(waveform, self.target_length, self.fill_value)
		return waveform


class Pad(WaveformTransform):
	def __init__(self, target_length: int, fill_value: float = 0.0, align: str = "left", p: float = 1.0):
		super().__init__(p)
		self.target_length = target_length
		self.fill_value = fill_value
		self.align = align

	def apply(self, waveform: Tensor) -> Tensor:
		waveform = pad(waveform, self.target_length, self.fill_value, self.align)
		return waveform


def pad_align_left(waveform: Tensor, target_length: int, fill_value: float) -> Tensor:
	""" Align to left by adding zeros to right. """
	if target_length > waveform.shape[-1]:
		missing = target_length - waveform.shape[-1]
		shape_zeros = list(waveform.shape[:-1]) + [missing]

		waveform = torch.cat((
			waveform,
			torch.full(shape_zeros, fill_value, dtype=waveform.dtype, device=waveform.device),
		), dim=-1)
	return waveform


def pad_align_right(waveform: Tensor, target_length: int, fill_value: float) -> Tensor:
	""" Align to right by adding zeros to left. """
	if target_length > waveform.shape[-1]:
		missing = target_length - waveform.shape[-1]
		shape_zeros = list(waveform.shape[:-1]) + [missing]

		waveform = torch.cat((
			torch.full(shape_zeros, fill_value, dtype=waveform.dtype, device=waveform.device),
			waveform
		), dim=-1)
	return waveform


def pad_align_center(waveform: Tensor, target_length: int, fill_value: float) -> Tensor:
	""" Align to center by adding half of zeros to left and the other half to right. """
	if target_length > waveform.shape[-1]:
		missing = target_length - waveform.shape[-1]

		missing_left = missing // 2 + missing % 2
		missing_right = missing // 2

		shape_zeros_left = list(waveform.shape[:-1]) + [missing_left]
		shape_zeros_right = list(waveform.shape[:-1]) + [missing_right]

		waveform = torch.cat((
			torch.full(shape_zeros_left, fill_value, dtype=waveform.dtype, device=waveform.device),
			waveform,
			torch.full(shape_zeros_right, fill_value, dtype=waveform.dtype, device=waveform.device)
		), dim=-1)
	return waveform


def pad_align_random(waveform: Tensor, target_length: int, fill_value: float) -> Tensor:
	""" Randomly add zeros to left and right for having the size of target_length. """
	if target_length > waveform.shape[-1]:
		missing = target_length - waveform.shape[-1]

		missing_left = torch.randint(low=0, high=missing, size=()).item()
		missing_right = missing - missing_left

		shape_zeros_left = list(waveform.shape[:-1]) + [missing_left]
		shape_zeros_right = list(waveform.shape[:-1]) + [missing_right]

		waveform = torch.cat((
			torch.full(shape_zeros_left, fill_value, dtype=waveform.dtype, device=waveform.device),
			waveform,
			torch.full(shape_zeros_right, fill_value, dtype=waveform.dtype, device=waveform.device),
		), dim=-1)
	return waveform


def pad(waveform: Tensor, target_length: int, fill_value: float, align: str) -> Tensor:
	if align == "left":
		return pad_align_left(waveform, target_length, fill_value)
	elif align == "right":
		return pad_align_right(waveform, target_length, fill_value)
	elif align == "center":
		return pad_align_center(waveform, target_length, fill_value)
	elif align == "random":
		return pad_align_random(waveform, target_length, fill_value)
	else:
		raise ValueError(f"Unknown alignment \"{align}\". Must be one of {str(['left', 'right', 'center', 'random'])}.")
