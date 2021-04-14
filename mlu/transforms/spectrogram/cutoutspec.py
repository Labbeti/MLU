
import logging
import math
import torch

from torch import Tensor
from typing import Tuple, Union

from mlu.transforms.base import SpectrogramTransform


class CutOutSpec(SpectrogramTransform):
	def __init__(
		self,
		freq_scales: Tuple[float, float] = (0.1, 0.5),
		time_scales: Tuple[float, float] = (0.1, 0.5),
		fill_value: float = -80.0,
		p: float = 1.0,
		freq_dim: int = -2,
		time_dim: int = -1,
		width_scales: Union[float, Tuple[float, float], None] = None,
		height_scales: Union[float, Tuple[float, float], None] = None,
	):
		"""
			CutOut transform for spectrogram tensors.

			Input must be of shape (..., freq, time).

			:param width_scales: The range of ratios of the rectangle cut width. (default: (0.1, 0.5))
			:param height_scales: The range of ratios of the rectangle cut height. (default: (0.1, 0.5))
			:param fill_value: The value used for fill. (default: -80.0)
			:param p: The probability to apply the transform. (default: 1.0)
			:param freq_dim: The dimension index of the spectrogram frequencies (default: -2)
			:param time_dim: The dimension index of the spectrogram time steps (default: -1)
		"""
		super().__init__(p=p)

		if width_scales is not None:
			logging.warning(
				'(package MLU): Parameter name "width_scales" is depreciated. Please use "freq_scales" instead.'
			)
			freq_scales = width_scales if isinstance(width_scales, tuple) else (width_scales, width_scales)

		if height_scales is not None:
			logging.warning(
				'(package MLU): Parameter name "height_scales" is depreciated. Please use "time_scales" instead.'
			)
			time_scales = height_scales if isinstance(height_scales, tuple) else (height_scales, height_scales)

		self.freq_scales = freq_scales
		self.time_scales = time_scales
		self.fill_value = fill_value
		self.freq_dim = freq_dim
		self.time_dim = time_dim

	def process(self, data: Tensor) -> Tensor:
		if len(data.shape) < 2:
			raise RuntimeError(
				f'Invalid data shape {data.shape}.'
				f'Must have at least 2 dimensions but found {len(data.shape)}.'
			)

		slices = [slice(None)] * len(data.shape)

		# Prepare slices indexes for frequencies and time dimensions
		slices[self.freq_dim] = self._gen_slice(data.shape[self.freq_dim], self.freq_scales)
		slices[self.time_dim] = self._gen_slice(data.shape[self.time_dim], self.time_scales)

		# Set are to fill_value
		data = data.clone()
		data[slices] = self.fill_value
		return data

	@staticmethod
	def _gen_slice(size: int, scales: Tuple[float, float]) -> slice:
		cutout_size_min = math.ceil(scales[0] * size)
		cutout_size_max = max(math.ceil(scales[1] * size), cutout_size_min + 1)
		cutout_size = torch.randint(cutout_size_min, cutout_size_max, ())

		cutout_start = torch.randint(0, max(size - cutout_size, 1), ())
		cutout_end = max(cutout_start + cutout_size, cutout_start + 1)

		return slice(cutout_start, cutout_end)
