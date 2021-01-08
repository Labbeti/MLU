
import torch

from mlu.metrics.classification.precision import Precision
from mlu.metrics.classification.recall import Recall
from mlu.metrics.base import Metric
from torch import Tensor


class FScore(Metric[Tensor, Tensor, Tensor]):
	def __init__(self):
		super().__init__()
		self.recall = Recall()
		self.precision = Precision()

	def compute_score(self, input_: Tensor, target: Tensor) -> Tensor:
		recall = self.recall(input_, target)
		precision = self.precision(input_, target)

		denominator = recall + precision
		if denominator != 0.0:
			score = precision * recall / denominator
		else:
			score = torch.zeros(1)

		return score
