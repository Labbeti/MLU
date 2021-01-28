
import torch

from mlu.transforms.base import WaveformTransform
from torch import Tensor
from typing import Tuple, Union


class Occlusion(WaveformTransform):
	def __init__(
		self,
		scales: Union[float, Tuple[float, float]] = 0.1,
		fill_value: float = 0.0,
		dim: int = -1,
		p: float = 1.0,
	):
		"""
			Occlusion waveform transform.

			:param scales: The scale of the occlusion size.
			:param fill_value: The fill value for occlusion.
			:param dim: The dimension to apply the occlusion.
			:param p: The probability to apply the transform.
		"""
		super().__init__(p)
		self.scales = scales
		self.fill_value = fill_value
		self.dim = dim

	def apply(self, waveform: Tensor) -> Tensor:
		length = waveform.shape[self.dim]
		if isinstance(self.scales, float):
			occlusion_size = round(self.scales * length)
		else:
			min_scale, max_scale = self.scales
			occlusion_min, occlusion_max = round(min_scale * length), round(max_scale * length)
			occlusion_size = torch.randint(low=occlusion_min, high=occlusion_max, size=()).item()

		start_max = max(length - occlusion_size, 1)
		start = torch.randint(low=0, high=start_max, size=()).item()
		end = start + occlusion_size

		slices = [slice(None)] * len(waveform.shape)
		slices[self.dim] = slice(start, end)
		waveform = waveform.clone()
		waveform[slices] = self.fill_value
		return waveform
