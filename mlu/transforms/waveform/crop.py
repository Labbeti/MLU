
import torch

from mlu.transforms.base import WaveformTransform

from torch import Tensor


class CropAlignLeft(WaveformTransform):
	def __init__(self, target_length: int, dim: int = -1, p: float = 1.0):
		super().__init__(p)
		self.target_length = target_length
		self.dim = dim

	def apply(self, waveform: Tensor) -> Tensor:
		waveform = crop_align_left(waveform, self.target_length, self.dim)
		return waveform


class CropAlignRight(WaveformTransform):
	def __init__(self, target_length: int, dim: int = -1, p: float = 1.0):
		super().__init__(p)
		self.target_length = target_length
		self.dim = dim

	def apply(self, waveform: Tensor) -> Tensor:
		waveform = crop_align_right(waveform, self.target_length, self.dim)
		return waveform


class CropAlignCenter(WaveformTransform):
	def __init__(self, target_length: int, dim: int = -1, p: float = 1.0):
		super().__init__(p)
		self.target_length = target_length
		self.dim = dim

	def apply(self, waveform: Tensor) -> Tensor:
		waveform = crop_align_center(waveform, self.target_length, self.dim)
		return waveform


class CropAlignRandom(WaveformTransform):
	def __init__(self, target_length: int, dim: int = -1, p: float = 1.0):
		super().__init__(p)
		self.target_length = target_length
		self.dim = dim

	def apply(self, waveform: Tensor) -> Tensor:
		waveform = crop_align_random(waveform, self.target_length, self.dim)
		return waveform


class Crop(WaveformTransform):
	def __init__(self, target_length: int, dim: int = -1, align: str = "left", p: float = 1.0):
		super().__init__(p)
		self.target_length = target_length
		self.dim = dim
		self.align = align

	def apply(self, waveform: Tensor) -> Tensor:
		waveform = crop(waveform, self.target_length, self.dim, self.align)
		return waveform


def crop_align_left(waveform: Tensor, target_length: int, dim: int) -> Tensor:
	""" Align to left by removing values from right. """
	if waveform.shape[dim] > target_length:
		slices = [slice(None)] * len(waveform.shape)
		slices[dim] = slice(target_length)
		waveform = waveform[slices]
	return waveform


def crop_align_right(waveform: Tensor, target_length: int, dim: int) -> Tensor:
	""" Align to right by removing values from left. """
	if waveform.shape[dim] > target_length:
		start = waveform.shape[dim] - target_length
		slices = [slice(None)] * len(waveform.shape)
		slices[dim] = slice(start, None)
		waveform = waveform[slices]
	return waveform


def crop_align_center(waveform: Tensor, target_length: int, dim: int) -> Tensor:
	""" Align to center by removing half of the values in left and the other half in right. """
	if waveform.shape[dim] > target_length:
		diff = waveform.shape[dim] - target_length
		start = diff // 2 + diff % 2
		end = start + target_length
		slices = [slice(None)] * len(waveform.shape)
		slices[dim] = slice(start, end)
		waveform = waveform[slices]
	return waveform


def crop_align_random(waveform: Tensor, target_length: int, dim: int) -> Tensor:
	""" Randomly remove values in left and right. """
	if waveform.shape[dim] > target_length:
		diff = waveform.shape[dim] - target_length
		start = torch.randint(low=0, high=diff, size=()).item()
		end = start + target_length
		slices = [slice(None)] * len(waveform.shape)
		slices[dim] = slice(start, end)
		waveform = waveform[slices]
	return waveform


def crop(waveform: Tensor, target_length: int, dim: int, align: str) -> Tensor:
	if align == "left":
		return crop_align_left(waveform, target_length, dim)
	elif align == "right":
		return crop_align_right(waveform, target_length, dim)
	elif align == "center":
		return crop_align_center(waveform, target_length, dim)
	elif align == "random":
		return crop_align_random(waveform, target_length, dim)
	else:
		raise ValueError(f"Unknown alignment \"{align}\". Must be one of {str(['left', 'right', 'center', 'random'])}.")
