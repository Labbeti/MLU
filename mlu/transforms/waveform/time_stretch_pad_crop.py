
from mlu.transforms.base import WaveformTransform
from mlu.transforms.waveform.crop import Crop
from mlu.transforms.waveform.pad import Pad
from mlu.transforms.waveform.time_stretch import TimeStretch

from torch import Tensor
from typing import Optional, Tuple, Union


class TimeStretchPadCrop(WaveformTransform):
	def __init__(
		self,
		rates: Union[Tuple[float, float], float] = (0.9, 1.1),
		target_length: Union[int, str] = 'auto',
		align: str = 'random',
		fill_value: float = 0.0,
		dim: int = -1,
		p: float = 1.0,
	):
		"""
			TimeStretch, Pad and Crop the signal.

			:param rates: The ratio of the signal used for resize. (default: (0.9, 1.1))
			:param target_length: Optional target length of the signal dimension.
				If 'auto', the output will have the same shape than the input.
				(default: 'auto')
			:param align: Alignment to use for cropping and padding. Can be 'left', 'right', 'center' or 'random'.
				(default: 'random')
			:param fill_value: The fill value when padding the waveform. (default: 0.0)
			:param dim: The dimension to stretch and pad or crop. (default: -1)
			:param p: The probability to apply the transform. (default: 1.0)
		"""
		super().__init__(p=p)
		self.rates = rates
		self._target_length = target_length
		self.align = align
		self.fill_value = fill_value
		self.dim = dim

		target_length = self.target_length if isinstance(self.target_length, int) else 1
		self.stretch = TimeStretch(rates, dim=dim)
		self.pad = Pad(target_length, align, fill_value, dim, mode='constant')
		self.crop = Crop(target_length, align, dim)

	def process(self, data: Tensor) -> Tensor:
		if self.target_length == 'auto':
			target_length = data.shape[self.dim]
			self.pad.target_length = target_length
			self.crop.target_length = target_length

		return self.crop(self.pad(self.stretch(data)))

	@property
	def target_length(self) -> Union[int, str]:
		return self._target_length
