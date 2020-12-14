
from mlu.nn.functional.labels import nums_to_smooth_onehot
from mlu.nn.functional.math import mish

from torch import Tensor
from torch.nn import Module
from typing import Optional


DEFAULT_EPSILON = 2e-20


class OneHot(Module):
	def __init__(self, nb_classes: int, smooth: Optional[float] = 0.0):
		"""
			Convert label to one-hot encoding.

			:param nb_classes: The number of classes in the dataset.
			:param smooth: The optional label smoothing coefficient parameter.
		"""
		super().__init__()
		self.nb_classes = nb_classes
		self.smooth = smooth if smooth is not None else 0.0

	def forward(self, x: Tensor) -> Tensor:
		return nums_to_smooth_onehot(x, self.nb_classes, self.smooth)


class Squeeze(Module):
	def forward(self, x: Tensor) -> Tensor:
		return x.squeeze()


class UnSqueeze(Module):
	def __init__(self, dim: int):
		super().__init__()
		self.dim = dim

	def forward(self, x: Tensor) -> Tensor:
		return x.unsqueeze(self.dim)


class Mish(Module):
	"""
		Mish class for apply mish function.
	"""
	def __init__(self):
		super().__init__()

	def forward(self, x: Tensor) -> Tensor:
		return mish(x)
