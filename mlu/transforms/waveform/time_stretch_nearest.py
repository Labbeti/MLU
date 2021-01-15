
import torch

from mlu.transforms.base import WaveformTransform
from torch import Tensor


class TimeStretchNearest(WaveformTransform):
	def __init__(self, orig_freq: int = 16000, new_freq: int = 16000, p: float = 1.0):
		super().__init__(p=p)
		self.orig_freq = orig_freq
		self.new_freq = new_freq

	def apply(self, data: Tensor) -> Tensor:
		length = data.shape[-1]
		indexes = torch.arange(start=0, end=length, step=self.orig_freq / self.new_freq)
		indexes = indexes.round().long().clamp(max=length - 1)
		slices = [slice(None)] * (len(data.shape) - 1) + [indexes]
		return data[slices]
