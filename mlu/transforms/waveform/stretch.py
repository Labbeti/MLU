
import torch

from mlu.transforms.base import WaveformTransform
from torch import Tensor
from torch.distributions import Uniform
from typing import Tuple, Union


class Stretch(WaveformTransform):
	def __init__(
		self,
		rates: Union[float, Tuple[float, float]] = (0.5, 1.5),
		dim: int = -1,
		p: float = 1.0,
	):
		"""
			Stretch transform. Resample an audio waveform with nearest value.

			:param rates: The rate of the stretch. Ex: use 2.0 for multiply the signal length by 2. (default: (0.5, 1.5))
			:param dim: The dimension to modify. (default: -1)
			:param p: The probability to apply the transform. (default: 1.0)
		"""
		super().__init__(p=p)
		self.rates = rates if isinstance(rates, tuple) else (rates, rates)
		self.dim = dim

	def process(self, data: Tensor) -> Tensor:
		length = data.shape[self.dim]
		sampler = Uniform(low=self.rates[0], high=self.rates[1])
		rate = sampler.sample().item()
		step = 1.0 / rate
		indexes = torch.arange(start=0, end=length, step=step)
		indexes = indexes.floor().long().clamp(max=length - 1)
		slices = [slice(None)] * len(data.shape)
		slices[self.dim] = indexes
		data = data[slices].contiguous()
		return data

	def set_rates(self, rates: Union[float, Tuple[float, float]]):
		self.rates = rates if isinstance(rates, tuple) else (rates, rates)
