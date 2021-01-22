
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
		self.rolls = rolls if isinstance(rolls, tuple) else (rolls, rolls)
		self.uniform = Uniform(low=self.rolls[0], high=self.rolls[1]) if isinstance(rolls, tuple) else None

	def apply(self, spectrogram: Tensor) -> Tensor:
		roll = self.uniform.sample().item() if self.uniform is not None else self.rolls[0]
		roll_size = round(spectrogram.shape[self.dim] * roll)
		spectrogram = spectrogram.roll(roll_size, dims=self.dim)
		return spectrogram
