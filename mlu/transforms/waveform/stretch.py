
import torch

from mlu.transforms.base import WaveformTransform
from torch import Tensor
from torch.distributions import Uniform
from typing import Tuple, Union


class StretchNearestFreq(WaveformTransform):
	def __init__(
		self,
		orig_freq: int = 16000,
		new_freq: int = 16000,
		dim: int = -1,
		p: float = 1.0,
	):
		"""
			StretchNearestFreq transform.

			:param orig_freq: The original freq of the signal.
			:param new_freq: The new freq of the signal.
			:param dim: The dimension to modify.
			:param p: The probability to apply the transform.
		"""
		super().__init__(p=p)
		self.orig_freq = orig_freq
		self.new_freq = new_freq
		self.dim = dim

	def apply(self, data: Tensor) -> Tensor:
		length = data.shape[self.dim]
		indexes = torch.arange(start=0, end=length, step=self.orig_freq / self.new_freq)
		indexes = indexes.floor().long().clamp(max=length - 1)
		slices = [slice(None)] * len(data.shape)
		slices[self.dim] = indexes
		return data[slices]


class StretchNearestRate(WaveformTransform):
	def __init__(
		self,
		rates: Union[float, Tuple[float, float]] = (0.5, 1.5),
		dim: int = -1,
		p: float = 1.0
	):
		"""
			StretchNearestRate transform.

			:param rates: The rate of the stretch. Ex: use 2.0 for multiply the signal length by 2.
			:param dim: The dimension to modify.
			:param p: The probability to apply the transform.
		"""
		super().__init__(p=p)
		self.dim = dim

		self._rates = rates if isinstance(rates, tuple) else (rates, rates)
		self._uniform = Uniform(low=self._rates[0], high=self._rates[1])

	def apply(self, data: Tensor) -> Tensor:
		length = data.shape[self.dim]
		rate = self._uniform.sample().item()
		indexes = torch.arange(start=0, end=length, step=1.0 / rate)
		indexes = indexes.floor().long().clamp(max=length - 1)
		slices = [slice(None)] * len(data.shape)
		slices[self.dim] = indexes
		return data[slices]
