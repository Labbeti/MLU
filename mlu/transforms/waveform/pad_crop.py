
from mlu.transforms.base import WaveformTransform
from mlu.transforms.waveform.crop import crop
from mlu.transforms.waveform.pad import pad
from torch import Tensor


class PadCrop(WaveformTransform):
	def __init__(
		self,
		target_length: int,
		fill_value: float = 0.0,
		dim: int = -1,
		align: str = "random",
		p: float = 1.0,
	):
		"""
			Composition of pad and crop transforms. Useful for preprocess length of waveforms to a specific length.

			:param target_length: The target length of the waveform.
			:param fill_value: The value used to pad the waveform. (default: 0.0)
			:param dim: The dimension to apply the padding. (default: -1)
			:param align: The waveform alignment. Determine if zeros will be added to left or right.
				(available alignment are : 'left', 'right', 'center' and 'random').
				(default: 'random')
			:param p: The probability to apply the transform. (default: 1.0)
		"""
		super().__init__(p)
		self.target_length = target_length
		self.fill_value = fill_value
		self.dim = dim
		self.align = align

	def process(self, waveform: Tensor) -> Tensor:
		waveform = pad_crop(waveform, self.target_length, self.fill_value, self.dim, self.align)
		return waveform

	def set_target_length(self, target_length: int):
		self.target_length = target_length


def pad_crop(waveform: Tensor, target_length: int, fill_value: float, dim: int, align: str) -> Tensor:
	waveform = pad(waveform, target_length, fill_value, dim, align)
	waveform = crop(waveform, target_length, dim, align)
	return waveform
