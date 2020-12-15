
from mlu.transforms.base import WaveformTransform
from mlu.transforms.waveform.pad_crop import PadCrop
from mlu.transforms.waveform.time_stretch_nearest import TimeStretchNearest

from torch import Tensor
from torch.distributions import Uniform
from typing import Tuple, Union


class StretchPadCrop(WaveformTransform[Tensor, Tensor]):
	def __init__(
		self,
		rate: Union[Tuple[float, float], float] = (0.9, 1.1),
		align: str = "random",
		p: float = 1.0,
	):
		"""
			:param rate: The ratio of the signal used for resize.
			:param align: Alignment to use for cropping or padding. Can be 'left', 'right', 'center' or 'random'.
			:param p: The probability to apply the augmentation.
		"""
		super().__init__(p=p)
		self.rate = rate

		self.uniform = Uniform(low=rate[0], high=rate[1]) if not isinstance(rate, float) else None
		self.time_stretch = TimeStretchNearest()
		self.pad_crop = PadCrop(target_length=0, align=align)

	def apply(self, x: Tensor) -> Tensor:
		data_length = x.shape[-1]
		self.time_stretch.orig_freq = data_length
		self.time_stretch.new_freq = round(data_length * self._get_rate())
		self.pad_crop.set_target_length(data_length)

		x = self.time_stretch(x)
		x = self.pad_crop(x)

		return x

	def _get_rate(self) -> float:
		if self.uniform is not None:
			self.rate = self.uniform.sample().item()
		return self.rate
