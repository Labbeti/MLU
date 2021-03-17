
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
		self.dim = dim

		self.rates = rates if isinstance(rates, tuple) else (rates, rates)

	def process(self, data: Tensor) -> Tensor:
		return stretch(data, self.rates, self.dim)

	def set_rates(self, rates: Union[float, Tuple[float, float]]):
		self.rates = rates if isinstance(rates, tuple) else (rates, rates)


class StretchFreq(WaveformTransform):
	def __init__(
		self,
		orig_freq: int = 16000,
		new_freq: int = 16000,
		dim: int = -1,
		p: float = 1.0,
	):
		"""
			StretchFreq transform. Resample an audio waveform with nearest value.

			:param orig_freq: The original freq of the signal. (default: 16000)
			:param new_freq: The new freq of the signal. (default: 16000)
			:param dim: The dimension to modify. (default: -1)
			:param p: The probability to apply the transform. (default: 1.0)
		"""
		super().__init__(p=p)
		self.orig_freq = orig_freq
		self.new_freq = new_freq
		self.dim = dim

	def process(self, data: Tensor) -> Tensor:
		return stretch_with_freq(data, self.orig_freq, self.new_freq, self.dim)


def stretch(
	data: Tensor,
	rates: Union[float, Tuple[float, float]] = (0.9, 1.1),
	dim: int = -1,
) -> Tensor:
	"""
		Stretch transform.

		:param data: The waveform Tensor to transform.
		:param rates: The rate of the stretch. Ex: use 2.0 for multiply the signal length by 2. (default: (0.5, 1.5))
		:param dim: The dimension to modify. (default: -1)
	"""
	length = data.shape[dim]
	sampler = Uniform(low=rates[0], high=rates[1])
	rate = sampler.sample().item()
	step = 1.0 / rate
	indexes = torch.arange(start=0, end=length, step=step)
	indexes = indexes.floor().long().clamp(max=length - 1)
	slices = [slice(None)] * len(data.shape)
	slices[dim] = indexes
	data = data[slices].contiguous()
	return data


def stretch_with_freq(
	data: Tensor,
	origin_freq: int = 16000,
	target_freq: int = 16000,
	dim: int = -1,
):
	rate = origin_freq / target_freq
	return stretch(data, rate, dim)
