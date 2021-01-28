
import torch

from mlu.metrics.base import Metric
from torch import Tensor
from typing import Callable, Optional


class BinaryAccuracy(Metric):
	def __init__(
		self,
		threshold_input: Optional[float] = None,
		threshold_target: Optional[float] = None,
		reduce_fn: Callable = torch.mean
	):
		super().__init__()
		self.threshold_input = threshold_input
		self.threshold_target = threshold_target
		self.reduce_fn = reduce_fn

	def compute_score(self, input_: Tensor, target: Tensor) -> Tensor:
		assert input_.shape == target.shape
		assert 0 <= len(input_.shape) <= 2

		if self.threshold_input is not None:
			input_ = input_.ge(self.threshold_input)

		if self.threshold_target is not None:
			target = target.ge(self.threshold_target)

		score = input_.eq(target).float()
		score = self.reduce_fn(score)
		return score