
from mlu.metrics.classification.precision import Precision
from mlu.metrics.classification.recall import Recall
from mlu.metrics.base import Metric
from torch import Tensor
from typing import Callable, Optional


class FScore(Metric):
	def __init__(self, dim: Optional[int] = 1, reduce_fn: Callable = lambda x: x):
		"""
			FScore metric. (micro).

			:param dim: The dimension to compute the score.
			:param reduce_fn: The reduction function to apply.
		"""
		super().__init__()
		self.reduce_fn = reduce_fn
		self.recall = Recall(dim, lambda x: x)
		self.precision = Precision(dim, lambda x: x)

	def compute_score(self, input_: Tensor, target: Tensor) -> Tensor:
		"""
			Compute score with one-hot or multi-hot inputs and targets.

			:param input_: Shape (nb classes) or (nb samplers, nb classes) binary tensor.
			:param target: Shape (nb classes) or (nb samplers, nb classes) binary tensor.
			:return: Score(s) as tensor in range [0, 1].
		"""
		recall = self.recall(input_, target)
		precision = self.precision(input_, target)

		score = 2.0 * precision * recall / (recall + precision)
		score[score.isnan()] = 0.0

		score = self.reduce_fn(score)
		return score
