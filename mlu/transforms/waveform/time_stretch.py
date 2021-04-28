
import logging
import torch

from torch import Tensor
from torch.distributions import Uniform
from typing import Tuple

from mlu.transforms.base import WaveformTransform


class TimeStretch(WaveformTransform):
	def __init__(
		self,
		rates: Tuple[float, float] = (0.5, 1.5),
		interpolation: str = 'nearest',
		dim: int = -1,
		p: float = 1.0,
	):
		"""
			Stretch transform. Resample an audio waveform with nearest value.

			:param rates: The rate of the stretch. Ex: use 2.0 for multiply the signal length by 2. (default: (0.5, 1.5))
			:param dim: The dimension to modify. (default: -1)
			:param p: The probability to apply the transform. (default: 1.0)
		"""
		if interpolation not in ('nearest',):
			raise ValueError(f'Invalid interpolation mode "{interpolation}". Must be one of ("nearest").')

		super().__init__(p=p)
		self.rates = rates
		self.interpolation = interpolation
		self.dim = dim

	def process(self, data: Tensor) -> Tensor:
		if isinstance(self.rates, tuple):
			sampler = Uniform(*self.rates)
			rate = sampler.sample().item()
		else:
			rate = self.rates

		if self.interpolation == 'nearest':
			data = self.stretch_nearest(data, rate)
		elif self.interpolation == 'linear':
			data = self.stretch_linear(data, rate)
		else:
			raise ValueError(f'Invalid interpolation mode "{self.interpolation}". Must be one of ("nearest", "linear").')

		return data

	def stretch_nearest(self, data: Tensor, rate: float) -> Tensor:
		length = data.shape[self.dim]
		step = 1.0 / rate
		indexes = torch.arange(0, length, step)
		indexes = indexes.round().long().clamp(max=length - 1)
		slices = [slice(None)] * len(data.shape)
		slices[self.dim] = indexes
		output = data[slices]
		return output.contiguous()

	def stretch_linear(self, data: Tensor, rate: float) -> Tensor:
		raise NotImplementedError('This mode is currently disabled.')
		# logging.warning('This mode is currently experimental.')
		#
		# length = data.shape[self.dim]
		# step = 1.0 / rate
		# indexes = torch.arange(0, length, step)
		#
		# indexes_floor = indexes.floor().long().clamp(max=length - 1)
		# slices_floor = [slice(None)] * len(data.shape)
		# slices_floor[self.dim] = indexes_floor
		# data_floor = data[slices_floor]
		#
		# indexes_ceil = indexes.ceil().long().clamp(max=length - 1)
		# slices_ceil = [slice(None)] * len(data.shape)
		# slices_ceil[self.dim] = indexes_ceil
		# data_ceil = data[slices_floor]
		#
		# coefficient_floor = indexes - indexes_floor
		# coefficient_floor[coefficient_floor == 0.0] = 1.0
		# coefficient_ceil = 1.0 - coefficient_floor
		#
		# output = data_floor * coefficient_floor + data_ceil * coefficient_ceil
		# return output.contiguous()
