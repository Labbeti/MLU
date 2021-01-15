
import torch

from mlu.transforms.base import WaveformTransform
from torch import Tensor
from torch.distributions import Uniform
from typing import Tuple, Union


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


class StretchNearestRate(WaveformTransform):
	def __init__(self, rates: Union[float, Tuple[float, float]] = (0.5, 1.5), p: float = 1.0):
		"""
			StretchNearestRate transform.
			TODO : test this transform

			:param rates:
			:param p: The probability to apply the transform.
		"""
		super().__init__(p=p)
		self.rates = rates if isinstance(rates, tuple) else (rates, rates)
		self.uniform = Uniform(low=self.rates[0], high=self.rates[1])

	def apply(self, data: Tensor) -> Tensor:
		rate = self.uniform.sample().item()
		length = data.shape[-1]
		indexes = torch.arange(start=0, end=length, step=1.0 / rate)
		indexes = indexes.round().long().clamp(max=length - 1)
		slices = [slice(None)] * (len(data.shape) - 1) + [indexes]
		return data[slices]
