
import torch

from torch import Tensor
from typing import Callable, Optional

from mlu.metrics.base import Metric


class Recall(Metric):
	def __init__(
		self,
		dim: Optional[int] = -1,
		threshold_input: Optional[float] = 0.5,
		threshold_target: Optional[float] = 0.5,
		reduce_fn: Callable = torch.mean
	):
		"""
			Compute Recall score between binary vectors.

			>>> # Recall = TP / (TP + FN) where TP = True Positives, FN = False Negatives.

			Vectors must be binary tensors of shape (nb classes) or (nb samplers, nb classes).

			:param dim: The dimension to compute the score. (default: -1)
			:param threshold_input: The threshold value for binarize input vectors. (default: 0.5)
			:param threshold_target: The threshold value for binarize target vectors. (default: 0.5)
			:param reduce_fn: The reduction function to apply. (default: torch.mean)
		"""
		super().__init__()
		self.dim = dim if dim is not None else ()
		self.threshold_input = threshold_input
		self.threshold_target = threshold_target
		self.reduce_fn = reduce_fn

	def compute_score(self, input_: Tensor, target: Tensor) -> Tensor:
		"""
			Compute score with one-hot or multi-hot inputs and targets.

			:param input_: Shape (nb classes) or (nb samplers, nb classes) binary tensor.
			:param target: Shape (nb classes) or (nb samplers, nb classes) binary tensor.
			:return: Score(s) as tensor in range [0, 1].
		"""
		assert input_.shape == target.shape, \
			f"Mismatch between shapes {str(input_.shape)} and {str(target.shape)} for Recall metric."

		if self.threshold_input is not None:
			input_ = input_.ge(self.threshold_input).float()

		if self.threshold_target is not None:
			target = target.ge(self.threshold_target).float()

		assert input_.eq(0.0).logical_or(input_.eq(1.0)).all(), "Input must be binary tensor containing only 0 and 1."
		assert target.eq(0.0).logical_or(target.eq(1.0)).all(), "Target must be binary tensor containing only 0 and 1."

		true_positives = (input_ * target).sum(dim=self.dim)
		# TODO : maybe simplify TP + FN = Possible positives, sum(target)
		false_negatives = (target - input_).ge(1.0).sum(dim=self.dim)
		score = true_positives / (true_positives + false_negatives)
		score[score.isnan()] = 0.0
		score = self.reduce_fn(score)

		return score
