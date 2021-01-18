
from mlu.transforms.base import SpectrogramTransform
from mlu.utils.misc import random_rect

from torch import Tensor
from typing import Tuple, Union


class CutOutSpec(SpectrogramTransform):
	def __init__(
		self,
		width_scale_range: Union[Tuple[float, float], float] = (0.1, 0.5),
		height_scale_range: Union[Tuple[float, float], float] = (0.1, 0.5),
		fill_value: float = -80.0,
		p: float = 1.0,
	):
		super().__init__(p=p)
		self.width_scale_range = width_scale_range \
			if isinstance(width_scale_range, tuple) else (width_scale_range, width_scale_range)
		self.height_scale_range = height_scale_range \
			if isinstance(height_scale_range, tuple) else (height_scale_range, height_scale_range)
		self.fill_value = fill_value

	def apply(self, x: Tensor) -> Tensor:
		assert len(x.shape) >= 2
		height, width = x.shape[-2], x.shape[-1]
		left, right, top, down = random_rect(height, width, self.width_scale_range, self.height_scale_range)
		slices = [slice(None)] * (len(x.shape) - 2) + [slice(left, right), slice(top, down)]

		out = x.clone()
		out[slices] = self.fill_value
		return out
