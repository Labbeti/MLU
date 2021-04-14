
import torch

from torch import Tensor
from typing import Callable, Optional

from mlu.metrics.base import Metric


class BinaryAccuracy(Metric):
	def __init__(
		self,
		threshold_input: Optional[float] = 0.5,
		threshold_target: Optional[float] = None,
		reduce_fn: Callable = torch.mean
	):
		"""
			Binary Accuracy metric.
			Compute the accuracy between two multihot vectors.

			:param threshold_input: The optional threshold to apply to inputs.
			:param threshold_target: The optional threshold to apply to targets.
			:param reduce_fn: The reduction function to apply to score.
		"""
		super().__init__()
		self.threshold_input = threshold_input
		self.threshold_target = threshold_target
		self.reduce_fn = reduce_fn

	def compute_score(self, pred: Tensor, target: Tensor) -> Tensor:
		assert pred.shape == target.shape
		assert 0 <= len(pred.shape) <= 2

		if self.threshold_input is not None:
			pred = pred.ge(self.threshold_input).float()

		if self.threshold_target is not None:
			target = target.ge(self.threshold_target).float()

		assert pred.eq(0.0).logical_or(pred.eq(1.0)).all(), 'Prediction must be binary tensor containing only 0 and 1.'
		assert target.eq(0.0).logical_or(target.eq(1.0)).all(), 'Target must be binary tensor containing only 0 and 1.'

		score = pred.eq(target).float()
		score = self.reduce_fn(score)
		return score
