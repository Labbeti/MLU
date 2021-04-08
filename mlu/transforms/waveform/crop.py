
import torch

from torch import Tensor
from mlu.transforms.base import WaveformTransform


class Crop(WaveformTransform):
	def __init__(self, target_length: int, align: str = "left", dim: int = -1, p: float = 1.0):
		super().__init__(p=p)
		self.target_length = target_length
		self.align = align
		self.dim = dim

	def process(self, waveform: Tensor) -> Tensor:
		if self.align == "center":
			return self.crop_align_center(waveform)
		elif self.align == "left":
			return self.crop_align_left(waveform)
		elif self.align == "random":
			return self.crop_align_random(waveform)
		elif self.align == "right":
			return self.crop_align_right(waveform)
		else:
			raise ValueError(f"Unknown alignment '{self.align}'. Must be one of {str(['left', 'right', 'center', 'random'])}.")

	def crop_align_center(self, waveform: Tensor) -> Tensor:
		if waveform.shape[self.dim] > self.target_length:
			diff = waveform.shape[self.dim] - self.target_length
			start = diff // 2 + diff % 2
			end = start + self.target_length
			slices = [slice(None)] * len(waveform.shape)
			slices[self.dim] = slice(start, end)
			waveform = waveform[slices]
		return waveform

	def crop_align_left(self, waveform: Tensor) -> Tensor:
		if waveform.shape[self.dim] > self.target_length:
			slices = [slice(None)] * len(waveform.shape)
			slices[self.dim] = slice(self.target_length)
			waveform = waveform[slices]
		return waveform

	def crop_align_random(self, waveform: Tensor) -> Tensor:
		if waveform.shape[self.dim] > self.target_length:
			diff = waveform.shape[self.dim] - self.target_length
			start = torch.randint(low=0, high=diff, size=()).item()
			end = start + self.target_length
			slices = [slice(None)] * len(waveform.shape)
			slices[self.dim] = slice(start, end)
			waveform = waveform[slices]
		return waveform

	def crop_align_right(self, waveform: Tensor) -> Tensor:
		if waveform.shape[self.dim] > self.target_length:
			start = waveform.shape[self.dim] - self.target_length
			slices = [slice(None)] * len(waveform.shape)
			slices[self.dim] = slice(start, None)
			waveform = waveform[slices]
		return waveform
