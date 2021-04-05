
import torch

from torch import Tensor
from typing import Callable, Optional

from mlu.metrics.classification.precision import Precision
from mlu.metrics.classification.recall import Recall
from mlu.metrics.base import Metric


class FScore(Metric):
	def __init__(
		self,
		dim: Optional[int] = -1,
		threshold_input: Optional[float] = 0.5,
		threshold_target: Optional[float] = 0.5,
		reduce_fn: Optional[Callable] = torch.mean,
	):
		"""
			FScore metric. (micro).

			>>> 'FScore = 2 * precision * recall / (recall + precision)'

			Vectors must be binary tensors of shape (nb classes) or (nb samplers, nb classes).

			:param dim: The dimension to compute the score. (default: -1)
			:param threshold_input: The threshold value for binarize input vectors. (default: 0.5)
			:param threshold_target: The threshold value for binarize target vectors. (default: 0.5)
			:param reduce_fn: The reduction function to apply. (default: torch.mean)
		"""
		super().__init__()
		self.threshold_input = threshold_input
		self.threshold_target = threshold_target
		self.reduce_fn = reduce_fn

		self.recall = Recall(dim, None, None, None)
		self.precision = Precision(dim, None, None, None)

	def compute_score(self, input_: Tensor, target: Tensor) -> Tensor:
		"""
			Compute score with one-hot or multi-hot inputs and targets.

			:param input_: Shape (nb classes) or (nb samplers, nb classes) binary tensor.
			:param target: Shape (nb classes) or (nb samplers, nb classes) binary tensor.
			:return: Score(s) as tensor in range [0, 1].
		"""
		if self.threshold_input is not None:
			input_ = input_.ge(self.threshold_input).float()

		if self.threshold_target is not None:
			target = target.ge(self.threshold_target).float()

		recall = self.recall(input_, target)
		precision = self.precision(input_, target)

		score = 2.0 * precision * recall / (recall + precision)
		score[score.isnan()] = 0.0

		if self.reduce_fn is not None:
			score = self.reduce_fn(score)

		return score
