
import torch

from mlu.metrics.base import Metric
from torch import Tensor
from typing import Callable, Optional


class Precision(Metric):
	def __init__(self, dim: Optional[int] = 1, reduce_fn: Callable = torch.mean):
		"""
			Compute Precision score between binary vectors.

			Recall = TP / (TP + FP) where TP = True Positives, FP = False Positives.
			Vectors must be 1D-tensors of shape (nb classes).

			:param dim: The dimension to compute the score.
			:param reduce_fn: The reduction function to apply.
		"""
		super().__init__()
		self.dim = dim if dim is not None else ()
		self.reduce_fn = reduce_fn

	def compute_score(self, input_: Tensor, target: Tensor) -> Tensor:
		"""
			Compute score with one-hot or multi-hot inputs and targets.

			:param input_: Shape (nb classes) or (nb samplers, nb classes) binary tensor.
			:param target: Shape (nb classes) or (nb samplers, nb classes) binary tensor.
			:return: Score(s) as tensor in range [0, 1].
		"""
		assert input_.shape == target.shape, \
			f"Mismatch between shapes {input_.shape} and {target.shape} for Precision metric."
		assert input_.eq(0.0).logical_or(input_.eq(1.0)).all(), "Input must be binary tensor."
		assert target.eq(0.0).logical_or(target.eq(1.0)).all(), "Target must be binary tensor."

		true_positives = (input_ * target).sum(dim=self.dim)
		false_positives = (input_ - target).ge(1.0).sum(dim=self.dim)
		score = true_positives / (true_positives + false_positives)
		score[score.isnan()] = 0.0
		score = self.reduce_fn(score)
		return score
