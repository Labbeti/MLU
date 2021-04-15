
import torch

from torch import Tensor
from torch.distributions.uniform import Uniform
from typing import Union

from mlu.transforms.base import Transform


class AdditiveNoise(Transform):
	def __init__(self, snr_db: float, clamp_max: Union[float, str, None] = 'auto', p: float = 1.0):
		super().__init__(p=p)
		self.snr_db = snr_db
		self.clamp_max = clamp_max

	def process(self, x: Tensor) -> Tensor:
		clamp_max = x.max() if self.clamp_max == 'auto' else self.clamp_max
		x = x + gen_noise(x, self.snr_db)
		if clamp_max is not None:
			x = torch.clamp(x, max=clamp_max)
		return x


class SubtractiveNoise(Transform):
	def __init__(self, snr_db: float, clamp_min: Union[float, str, None] = 'auto', p: float = 1.0):
		super().__init__(p=p)
		self.snr_db = snr_db
		self.clamp_min = clamp_min

	def process(self, x: Tensor) -> Tensor:
		clamp_min = x.min() if self.clamp_min == 'auto' else self.clamp_min
		x = x - gen_noise(x, self.snr_db)
		if clamp_min is not None:
			x = torch.clamp(x, min=clamp_min)
		return x


def gen_noise(x: Tensor, snr_db: float) -> Tensor:
	mean_sq_x = (x ** 2).mean()
	snr = 10 ** (snr_db / 10)
	high = torch.sqrt(3.0 * mean_sq_x / snr)
	uniform = Uniform(low=0, high=high)
	noise = uniform.sample(x.shape)
	return noise
