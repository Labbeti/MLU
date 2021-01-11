
import torch

from mlu.transforms.base import ImageTransform
from mlu.utils.misc import random_rect

from torch import Tensor
from typing import List, Tuple, Union


class Normalize(ImageTransform):
	"""
		Normalize an image in [0, 1].
	"""
	def __init__(
		self,
		source_range: Tuple[Union[float, int], Union[float, int]] = (0, 255),
		target_range: Tuple[Union[float, int], Union[float, int]] = (0, 1),
		p: float = 1.0
	):
		super().__init__(p=p)
		self.source_range = source_range
		self.target_range = target_range

	def apply(self, x: Tensor) -> Tensor:
		normalized = (x - self.source_range[0]) / (self.source_range[1] - self.source_range[0])
		return normalized * (self.target_range[1] - self.target_range[0]) + self.target_range[0]


class Standardize(ImageTransform):
	"""
		Standardize image with a list of means and standard-deviations.
	"""
	def __init__(self, means: List[float], stds: List[float], dim_channel: int = 2, p: float = 1.0):
		super().__init__(p=p)
		self.means = means
		self.stds = stds
		self.dim_channel = dim_channel

		if len(means) != len(stds):
			raise RuntimeError("Means and stds lists must have the same size.")

	def apply(self, x: Tensor) -> Tensor:
		output = torch.zeros_like(x)

		for channel, (mean, std) in enumerate(zip(self.means, self.stds)):
			slices = [slice(None) if j != self.dim_channel else channel for j in range(len(x.shape))]
			output[slices] = (x[slices] - mean) / std
		return output


class Gray(ImageTransform):
	"""
		Convert image to gray.
	"""
	def __init__(self, dim_channel: int = 2, p: float = 1.0):
		super().__init__(p=p)
		self.dim_channel = dim_channel

	def apply(self, x: Tensor) -> Tensor:
		nb_channels = x.shape[self.dim_channel]
		output = x.mean(dim=self.dim_channel)
		output = output.repeat([nb_channels] + [1] * (len(x.shape) - 1))
		output = output.permute(list(range(1, len(x.shape))) + [0])
		return output


class CutOutImg(ImageTransform):
	"""
		Put gray value in an area randomly placed.
	"""
	def __init__(
		self,
		width_scale_range: Tuple[float, float] = (0.1, 0.5),
		height_scale_range: Tuple[float, float] = (0.1, 0.5),
		fill_value: Union[float, int] = 0,
		p: float = 1.0,
	):
		super().__init__(p=p)
		self.width_scale_range = width_scale_range
		self.height_scale_range = height_scale_range
		self.fill_value = fill_value

	def apply(self, x: Tensor) -> Tensor:
		assert len(x.shape) == 3

		width, height = x.shape[0], x.shape[1]
		left, right, top, down = random_rect(width, height, self.width_scale_range, self.height_scale_range)

		output = x.clone()
		slices = [slice(left, right), slice(top, down), slice(None)]
		output[slices] = self.fill_value
		return output


class UniColor(ImageTransform):
	"""
		Delete 2 random channels in image for getting only 1 color.
	"""
	def __init__(self, minimal_value: Union[float, int] = 0, dim_channel: int = 2, p: float = 1.0):
		super().__init__(p=p)
		self.minimal_value = minimal_value
		self.dim_channel = dim_channel

	def apply(self, x: Tensor) -> Tensor:
		output = torch.full_like(x, fill_value=self.minimal_value)
		channel_random = torch.randint(low=0, high=len(x.shape), size=())
		output[channel_random] = x.max(dim=self.dim_channel)
		return output


class Inversion(ImageTransform):
	"""
		Invert pixel colors of an image.
	"""
	def __init__(self, max_value: Union[float, int] = 255, p: float = 1.0):
		super().__init__(p=p)
		self.max_value = max_value

	def apply(self, x: Tensor) -> Tensor:
		return self.max_value - x
