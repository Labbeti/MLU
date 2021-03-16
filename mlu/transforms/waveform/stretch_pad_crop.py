
from mlu.transforms.base import WaveformTransform
from mlu.transforms.waveform.pad_crop import PadCrop
from mlu.transforms.waveform.stretch import StretchNearestFreq, StretchNearestRate

from torch import Tensor
from torch.distributions import Uniform
from typing import Tuple, Optional, Union


class StretchPadCrop(WaveformTransform):
	def __init__(
		self,
		rates: Union[Tuple[float, float], float] = (0.9, 1.1),
		align: str = "random",
		dim: int = -1,
		p: float = 1.0,
	):
		"""
			Stretch, pad and crop the signal. The output will have the same shape than input.

			:param rates: The ratio of the signal used for resize. (default: (0.9, 1.1))
			:param align: Alignment to use for cropping and padding. Can be 'left', 'right', 'center' or 'random'.
				(default: 'random')
			:param dim: The dimension to stretch and pad or crop. (default: -1)
			:param p: The probability to apply the transform. (default: 1.0)
		"""
		super().__init__(p=p)
		self._rates = rates if isinstance(rates, tuple) else (rates, rates)
		self._dim = dim

		self._uniform = Uniform(low=self._rates[0], high=self._rates[1])
		self._stretch = StretchNearestFreq(dim=dim)
		self._pad_crop = PadCrop(target_length=0, fill_value=0.0, align=align, dim=dim)
		self._last_rate = None

	def process(self, x: Tensor) -> Tensor:
		length = x.shape[self._dim]
		self._stretch.orig_freq = length
		self._last_rate = self._uniform.sample().item()
		self._stretch.new_freq = round(length * self._last_rate)
		self._pad_crop.set_target_length(length)

		x = self._stretch(x)
		x = self._pad_crop(x)

		return x

	def prev_rate(self) -> Optional[float]:
		return self._last_rate


class StretchPadCropNew(WaveformTransform):
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
		self._dim = dim
		self._stretch = StretchNearestRate(rates=rates, dim=dim)
		self._pad_crop = PadCrop(target_length=0, fill_value=fill_value, align=align, dim=dim)

	def process(self, x: Tensor) -> Tensor:
		length = x.shape[self._dim]
		self._pad_crop.set_target_length(length)

		x = self._stretch(x)
		x = self._pad_crop(x)

		return x

	def prev_rate(self) -> Optional[float]:
		return self._stretch.prev_rate()
