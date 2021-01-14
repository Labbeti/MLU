
import torch

from mlu.transforms.base import WaveformTransform
from torch import Tensor
from typing import Tuple, Union


class Occlusion(WaveformTransform):
	"""
		Occlusion waveform transform.
	"""
	def __init__(self, scale: Union[float, Tuple[float, float]] = 0.1, fill_value: float = 0.0, p: float = 1.0):
		super().__init__(p)
		self.scale = scale
		self.fill_value = fill_value

	def apply(self, waveform: Tensor) -> Tensor:
		length = waveform.shape[-1]
		if isinstance(self.scale, float):
			occlusion_size = length * self.scale
		else:
			min_scale, max_scale = self.scale
			low, high = round(min_scale * length), round(max_scale * length)
			occlusion_size = torch.randint(low=low, high=high, size=()).item()

		start = torch.randint(low=0, high=length - occlusion_size, size=()).item()
		end = torch.randint(low=occlusion_size, high=length, size=()).item()

		slices = [slice(None)] * (len(waveform.shape) - 1) + [slice(start, end)]
		waveform[slices] = self.fill_value
		return waveform
