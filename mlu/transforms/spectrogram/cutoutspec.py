
from mlu.transforms.base import SpectrogramTransform
from mlu.utils.misc import random_rect

from torch import Tensor
from typing import Tuple, Union


class CutOutSpec(SpectrogramTransform):
	def __init__(
		self,
		width_scales: Union[Tuple[float, float], float] = (0.1, 0.5),
		height_scales: Union[Tuple[float, float], float] = (0.1, 0.5),
		fill_value: float = -80.0,
		p: float = 1.0,
	):
		"""
			CutOut transforms for spectrogram.

			Input must be of shape (..., freq, time).

			:param width_scales: The range of ratios of the rectangle cut width. (default: (0.1, 0.5))
			:param height_scales: The range of ratios of the rectangle cut height. (default: (0.1, 0.5))
			:param fill_value: The value used for fill. (default: -80.0)
			:param p: The probability to apply the transform. (default: 1.0)
		"""
		super().__init__(p=p)
		self.width_scale_range = width_scales \
			if isinstance(width_scales, tuple) else (width_scales, width_scales)
		self.height_scale_range = height_scales \
			if isinstance(height_scales, tuple) else (height_scales, height_scales)
		self.fill_value = fill_value

	def apply(self, spectrogram: Tensor) -> Tensor:
		assert len(spectrogram.shape) >= 2

		height, width = spectrogram.shape[-2], spectrogram.shape[-1]
		left, right, top, down = random_rect(height, width, self.width_scale_range, self.height_scale_range)
		slices = [slice(None)] * (len(spectrogram.shape) - 2) + [slice(left, right), slice(top, down)]
		spectrogram = spectrogram.clone()
		spectrogram[slices] = self.fill_value
		return spectrogram
