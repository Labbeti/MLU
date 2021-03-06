
import torch

from torch import Tensor
from typing import Callable, Optional

from mlu.metrics.base import Metric


class CategoricalAccuracy(Metric):
	def __init__(
		self,
		dim: int = -1,
		vector_input: bool = True,
		vector_target: bool = True,
		reduce_fn: Optional[Callable] = torch.mean
	):
		"""
			Compute the categorical accuracy between a batch of prediction and labels.

			:param dim: The dimension to compute the score. (default: -1)
			:param vector_input: If True, considers inputs as a vector of probabilities.
				If False, it will be considered as a vectors of classes index. (default: True)
			:param vector_target: If True, considers target as a vector of probabilities.
				If False, it will be considered as a vectors of classes index. (default: True)
			:param reduce_fn: The reduction function to apply. (default: torch.mean)
		"""
		super().__init__()
		self.dim = dim
		self.vector_input = vector_input
		self.vector_target = vector_target
		self.reduce_fn = reduce_fn

	def compute_score(self, input_: Tensor, target: Tensor) -> Tensor:
		if self.vector_input:
			input_ = input_.argmax(dim=self.dim)
		if self.vector_target:
			target = target.argmax(dim=self.dim)

		assert input_.shape == target.shape, "Input and target must have the same shape."
		assert 0 <= len(input_.shape) <= 2

		score = input_.eq(target).float()
		if self.reduce_fn is not None:
			score = self.reduce_fn(score)
		return score
