
from mlu.transforms.base import WaveformTransform
from mlu.transforms.waveform.pad_crop import pad_crop
from mlu.transforms.waveform.stretch import stretch

from torch import Tensor
from typing import Tuple, Union


class StretchPadCrop(WaveformTransform):
	def __init__(
		self,
		rates: Union[Tuple[float, float], float] = (0.9, 1.1),
		fill_value: float = 0.0,
		align: str = "random",
		dim: int = -1,
		p: float = 1.0,
	):
		"""
			Stretch, pad and crop the signal. The output will have the same shape than input.

			:param rates: The ratio of the signal used for resize. (default: (0.9, 1.1))
			:param fill_value: The fill value when padding the waveform. (default: 0.0)
			:param align: Alignment to use for cropping and padding. Can be 'left', 'right', 'center' or 'random'.
				(default: 'random')
			:param dim: The dimension to stretch and pad or crop. (default: -1)
			:param p: The probability to apply the transform. (default: 1.0)
		"""
		super().__init__(p=p)
		self.rates = rates
		self.fill_value = fill_value
		self.align = align
		self.dim = dim

	def process(self, x: Tensor) -> Tensor:
		return stretch_pad_crop(x, self.rates, self.fill_value, self.align, self.dim)


def stretch_pad_crop(
	data: Tensor,
	rates: Union[float, Tuple[float, float]] = (0.9, 1.1),
	fill_value: float = 0.0,
	align: str = "random",
	dim: int = -1,
) -> Tensor:
	"""
		Stretch, pad and crop the signal. The output will have the same shape than input.

		:param data: The waveform Tensor to transform.
		:param rates: The ratio of the signal used for resize. (default: (0.9, 1.1))
		:param fill_value: The fill value when padding the waveform. (default: 0.0)
		:param align: Alignment to use for cropping and padding. Can be 'left', 'right', 'center' or 'random'.
			(default: 'random')
		:param dim: The dimension to stretch and pad or crop. (default: -1)
	"""
	target_length = data.shape[dim]
	data = stretch(data, rates, dim)
	data = pad_crop(data, target_length, fill_value, dim, align)
	return data
