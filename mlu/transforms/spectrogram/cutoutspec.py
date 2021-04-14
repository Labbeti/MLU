
import math
import torch

from torch import Tensor
from torch.distributions import Uniform
from typing import Tuple, Union

from mlu.transforms.base import SpectrogramTransform


class CutOutSpec(SpectrogramTransform):
	def __init__(
		self,
		freq_scales: Tuple[float, float] = (0.1, 0.5),
		time_scales: Tuple[float, float] = (0.1, 0.5),
		fill_value: Union[float, Tuple[float, float]] = -100.0,
		same_across_channels: bool = True,
		freq_dim: int = -2,
		time_dim: int = -1,
		p: float = 1.0,
	):
		"""
			CutOut transform for spectrogram tensors.

			Input must be of shape (..., freq, time).

			Example :

			>>> from mlu.transforms import CutOutSpec
			>>> spectrogram = ...
			>>> augment = CutOutSpec((0.5, 0.5), (0.5, 0.5))
			>>> # Cut 25% of the spectrogram values
			>>> spectrogram_augmented = augment(spectrogram)

			:param freq_scales: The range of ratios for the frequencies dim. (default: (0.1, 0.5))
			:param time_scales: The range of ratios for the time steps dim. (default: (0.1, 0.5))
			:param fill_value: The value used for fill. (default: -100.0)
			:param freq_dim: The dimension index of the spectrogram frequencies (default: -2)
			:param time_dim: The dimension index of the spectrogram time steps (default: -1)
			:param p: The probability to apply the transform. (default: 1.0)
		"""
		if freq_dim == time_dim:
			raise ValueError('Frequency dimension index cannot be the same than time dimension index.')

		super().__init__(p=p)

		self.freq_scales = freq_scales
		self.time_scales = time_scales
		self.fill_value = fill_value
		self.same_across_channels = same_across_channels
		self.freq_dim = freq_dim
		self.time_dim = time_dim

	def process(self, data: Tensor) -> Tensor:
		if not isinstance(data, Tensor) or len(data.shape) < 2:
			raise RuntimeError(
				f'Input data must be a pytorch Tensor with at least 2 dimensions for CutOutSpec transform, '
				f'found {type(data)}' + (f' of shape {data.shape}' if hasattr(data, 'shape') else '') + '.'
			)

		data = data.clone()

		if self.same_across_channels:
			data = self._process_same_across_channels(data)
		else:
			n_dims = len(data.shape)
			if n_dims != 3 or self.freq_dim != -2 or self.time_dim != -1:
				raise NotImplementedError(
					f'When same_across_channels=False, n_dims must be equal to 3, freq_dim to -2 and time_dim to -1. '
					f'(n_dims={n_dims}, freq_dim={self.freq_dim}, time_dim={self.time_dim})'
				)

			batch_dim = 0
			indexes: list = [slice(None) for _ in range(n_dims)]
			for i in range(data.shape[batch_dim]):
				indexes[batch_dim] = i
				data[indexes] = self._process_same_across_channels(data[indexes])

		return data

	def _process_same_across_channels(self, data: Tensor) -> Tensor:
		# Prepare slices indexes for frequencies and time dimensions
		slices = [slice(None)] * len(data.shape)

		freq_start, freq_end, freq_size = self._gen_range(data.shape[self.freq_dim], self.freq_scales)
		time_start, time_end, time_size = self._gen_range(data.shape[self.time_dim], self.time_scales)
		slices[self.freq_dim] = slice(freq_start, freq_end)
		slices[self.time_dim] = slice(time_start, time_end)

		if isinstance(self.fill_value, tuple):
			uniform = Uniform(low=self.fill_value[0], high=self.fill_value[1])
			shape = torch.ones(len(data.shape), dtype=torch.int)
			shape[self.freq_dim] = freq_size
			shape[self.time_dim] = time_size
			fill_value = uniform.sample(shape.tolist())
		else:
			fill_value = self.fill_value

		# Set are to fill_value
		data[slices] = fill_value
		return data

	@staticmethod
	def _gen_range(size: int, scales: Tuple[float, float]) -> Tuple[int, int, int]:
		cutout_size_min = math.ceil(scales[0] * size)
		cutout_size_max = max(math.ceil(scales[1] * size), cutout_size_min + 1)
		cutout_size = torch.randint(cutout_size_min, cutout_size_max, ())

		cutout_start = torch.randint(0, max(size - cutout_size + 1, 1), ())
		cutout_end = cutout_start + cutout_size
		assert cutout_end - cutout_start == cutout_size, f'{cutout_end} - {cutout_start} != {cutout_size}'

		return cutout_start, cutout_end, cutout_size
