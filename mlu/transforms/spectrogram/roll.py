
from mlu.transforms.base import SpectrogramTransform
from torch import Tensor


class RollSpec(SpectrogramTransform[Tensor, Tensor]):
	def __init__(self, roll: float = 0.1, p: float = 1.0):
		super().__init__(p=p)
		self.roll = roll

	def apply(self, x: Tensor) -> Tensor:
		roll_size = round(x.shape[0] * self.roll)
		x = x.roll(roll_size, dims=0)
		return x
