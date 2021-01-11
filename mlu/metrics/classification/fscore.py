
import torch

from mlu.metrics.classification.precision import Precision
from mlu.metrics.classification.recall import Recall
from mlu.metrics.base import Metric
from torch import Tensor
from typing import Callable


class FScore(Metric[Tensor, Tensor, Tensor]):
	def __init__(self, dim: int = 1, reduce_fn: Callable = torch.mean):
		super().__init__()
		self.reduce_fn = reduce_fn
		self.recall = Recall(dim, lambda x: x)
		self.precision = Precision(dim, lambda x: x)

	def compute_score(self, input_: Tensor, target: Tensor) -> Tensor:
		recall = self.recall(input_, target)
		precision = self.precision(input_, target)

		denominator = recall + precision
		score = precision * recall / denominator
		score[denominator == 0.0] = 0.0

		score = self.reduce_fn(score)
		return score
