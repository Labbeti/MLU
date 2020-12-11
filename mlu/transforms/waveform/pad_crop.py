
from mlu.transforms.base import WaveformTransform
from mlu.transforms.waveform.crop import CropLeft, CropRight, CropCenter, CropRandom
from mlu.transforms.waveform.pad import PadLeft, PadRight, PadCenter, PadRandom
from torch import Tensor


class PadCrop(WaveformTransform[Tensor, Tensor]):
	def __init__(self, target_length: int, fill_value: float = 0.0, align: str = "random", p: float = 1.0):
		super().__init__(p)

		if align == "left":
			self.crop = CropLeft(target_length)
			self.pad = PadLeft(target_length, fill_value)
		elif align == "right":
			self.crop = CropRight(target_length)
			self.pad = PadRight(target_length, fill_value)
		elif align == "center":
			self.crop = CropCenter(target_length)
			self.pad = PadCenter(target_length, fill_value)
		elif align == "random":
			self.crop = CropRandom(target_length)
			self.pad = PadRandom(target_length, fill_value)

	def apply(self, waveform: Tensor) -> Tensor:
		waveform = self.pad(self.crop(waveform))
		return waveform

	def set_target_length(self, target_length: int):
		self.crop.target_length = target_length
		self.pad.target_length = target_length
