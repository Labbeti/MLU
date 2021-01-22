
import torch

from mlu.transforms.base import WaveformTransform

from torch import Tensor


class CropAlignLeft(WaveformTransform):
	def __init__(self, target_length: int, p: float = 1.0):
		super().__init__(p)
		self.target_length = target_length

	def apply(self, waveform: Tensor) -> Tensor:
		waveform = crop_align_left(waveform, self.target_length)
		return waveform


class CropAlignRight(WaveformTransform):
	def __init__(self, target_length: int, p: float = 1.0):
		super().__init__(p)
		self.target_length = target_length

	def apply(self, waveform: Tensor) -> Tensor:
		waveform = crop_align_right(waveform, self.target_length)
		return waveform


class CropAlignCenter(WaveformTransform):
	def __init__(self, target_length: int, p: float = 1.0):
		super().__init__(p)
		self.target_length = target_length

	def apply(self, waveform: Tensor) -> Tensor:
		waveform = crop_align_center(waveform, self.target_length)
		return waveform


class CropAlignRandom(WaveformTransform):
	def __init__(self, target_length: int, p: float = 1.0):
		super().__init__(p)
		self.target_length = target_length

	def apply(self, waveform: Tensor) -> Tensor:
		waveform = crop_align_random(waveform, self.target_length)
		return waveform


class Crop(WaveformTransform):
	def __init__(self, target_length: int, align: str = "left", p: float = 1.0):
		super().__init__(p)
		self.target_length = target_length
		self.align = align

	def apply(self, waveform: Tensor) -> Tensor:
		waveform = crop(waveform, self.target_length, self.align)
		return waveform


def crop_align_left(waveform: Tensor, target_length: int) -> Tensor:
	""" Align to left by removing values from right. """
	if waveform.shape[-1] > target_length:
		slices = [slice(None)] * (len(waveform.shape) - 1) + [slice(target_length)]
		waveform = waveform[slices]
	return waveform


def crop_align_right(waveform: Tensor, target_length: int) -> Tensor:
	""" Align to right by removing values from left. """
	if waveform.shape[-1] > target_length:
		start = waveform.shape[-1] - target_length
		slices = [slice(None)] * (len(waveform.shape) - 1) + [slice(start, None)]
		waveform = waveform[slices]
	return waveform


def crop_align_center(waveform: Tensor, target_length: int) -> Tensor:
	""" Align to center by removing half of the values in left and the other half in right. """
	if waveform.shape[-1] > target_length:
		diff = waveform.shape[-1] - target_length
		start = diff // 2 + diff % 2
		end = start + target_length
		slices = [slice(None)] * (len(waveform.shape) - 1) + [slice(start, end)]
		waveform = waveform[slices]
	return waveform


def crop_align_random(waveform: Tensor, target_length: int) -> Tensor:
	""" Randomly remove values in left and right. """
	if waveform.shape[-1] > target_length:
		diff = waveform.shape[-1] - target_length
		start = torch.randint(low=0, high=diff, size=()).item()
		end = start + target_length
		slices = [slice(None)] * (len(waveform.shape) - 1) + [slice(start, end)]
		waveform = waveform[slices]
	return waveform


def crop(waveform: Tensor, target_length: int, align: str) -> Tensor:
	if align == "left":
		return crop_align_left(waveform, target_length)
	elif align == "right":
		return crop_align_right(waveform, target_length)
	elif align == "center":
		return crop_align_center(waveform, target_length)
	elif align == "random":
		return crop_align_random(waveform, target_length)
	else:
		raise ValueError(f"Unknown alignment \"{align}\". Must be one of {str(['left', 'right', 'center', 'random'])}.")
