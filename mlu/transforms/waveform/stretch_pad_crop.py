
from mlu.transforms.base import WaveformTransform
from mlu.transforms.waveform.pad_crop import PadCrop
from mlu.transforms.waveform.stretch import StretchNearestFreq

from torch import Tensor
from torch.distributions import Uniform
from typing import Tuple, Union


class StretchPadCrop(WaveformTransform):
	def __init__(
		self,
		rates: Union[Tuple[float, float], float] = (0.9, 1.1),
		align: str = "random",
		dim: int = -1,
		p: float = 1.0,
	):
		"""
			Stretch, pad and crop the signal.

			:param rates: The ratio of the signal used for resize.
			:param align: Alignment to use for cropping or padding. Can be 'left', 'right', 'center' or 'random'.
			:param p: The probability to apply the augmentation.
		"""
		super().__init__(p=p)
		self._rates = rates if isinstance(rates, tuple) else (rates, rates)

		self._uniform = Uniform(low=rates[0], high=rates[1])
		self._stretch = StretchNearestFreq(dim=dim)
		self._pad_crop = PadCrop(target_length=0, align=align, dim=dim)

	def apply(self, x: Tensor) -> Tensor:
		length = x.shape[self._dim]
		self._stretch.orig_freq = length
		self._stretch.new_freq = round(length * self._uniform.sample().item())
		self._pad_crop.set_target_length(length)

		x = self._stretch(x)
		x = self._pad_crop(x)

		return x
