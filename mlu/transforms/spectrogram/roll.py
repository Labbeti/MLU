
from mlu.transforms.base import SpectrogramTransform

from torch import Tensor
from torch.distributions.uniform import Uniform
from typing import Tuple, Union


class RollSpec(SpectrogramTransform):
	def __init__(
		self,
		dim: int,
		rolls: Union[float, Tuple[float, float]] = 0.1,
		p: float = 1.0,
	):
		super().__init__(p=p)
		self.dim = dim
		self.rolls = rolls

	def process(self, data: Tensor) -> Tensor:
		if isinstance(self.rolls, tuple):
			uniform = Uniform(*self.rolls)
			roll_scale = uniform.sample().item()
		else:
			roll_scale = self.rolls

		roll_size = round(data.shape[self.dim] * roll_scale)
		data = data.roll(roll_size, dims=self.dim)
		return data
