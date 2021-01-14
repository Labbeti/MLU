
from mlu.transforms.base import SpectrogramTransform
from mlu.utils.misc import random_rect

from torch import Tensor
from typing import Tuple


class CutOutSpec(SpectrogramTransform):
	def __init__(
		self,
		width_scale_range: Tuple[float, float] = (0.1, 0.5),
		height_scale_range: Tuple[float, float] = (0.1, 0.5),
		fill_value: float = -80.0,
		p: float = 1.0,
	):
		super().__init__(p)
		self.width_scale_range = width_scale_range
		self.height_scale_range = height_scale_range
		self.fill_value = fill_value

	def apply(self, x: Tensor) -> Tensor:
		width, height = x.shape[0], x.shape[1]
		left, right, top, down = random_rect(width, height, self.width_scale_range, self.height_scale_range)
		x[left:right, top:down] = self.fill_value
		return x
