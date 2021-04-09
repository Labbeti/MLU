
import torch

from mlu.transforms.base import SpectrogramTransform

from torch import Tensor
from typing import Tuple, Union


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
			print("WARNING (package MLU): Parameter name 'width_scales' is depreciated. Please use 'freq_scales' instead.")
			freq_scales = width_scales if isinstance(width_scales, tuple) else (width_scales, width_scales)

		if height_scales is not None:
			print("WARNING (package MLU): Parameter name 'height_scales' is depreciated. Please use 'time_scales' instead.")
			time_scales = height_scales if isinstance(height_scales, tuple) else (height_scales, height_scales)

		self.freq_scales = freq_scales
		self.time_scales = time_scales
		self.fill_value = fill_value
		self.freq_dim = freq_dim
		self.time_dim = time_dim

	def process(self, spectrogram: Tensor) -> Tensor:
		if len(spectrogram.shape) < 2:
			raise RuntimeError(
				f"Invalid spectrogram shape '{spectrogram.shape}."
				f"Must have at least 2 dimensions but found {len(spectrogram.shape)}."
			)

		slices = [slice(None)] * len(spectrogram.shape)

		# Prepare slices indexes for frequencies and time dimensions
		for dim, scales in zip([self.freq_dim, self.time_dim], [self.freq_scales, self.time_scales]):
			size = spectrogram.shape[dim]

			cutout_size_min = round(scales[0] * size)
			cutout_size_max = max(round(scales[1] * size), cutout_size_min + 1)
			cutout_size = torch.randint(cutout_size_min, cutout_size_max, ())

			cutout_start = torch.randint(0, max(size - cutout_size, 1), ())
			cutout_end = max(cutout_start + cutout_size, cutout_start + 1)

			slices[dim] = slice(cutout_start, cutout_end)

		# Set are to fill_value
		spectrogram = spectrogram.clone()
		spectrogram[slices] = self.fill_value
		return spectrogram
