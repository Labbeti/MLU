
import torch

from torch import Tensor
from typing import Callable
from mlu.metrics.base import Metric


class MSE(Metric):
	def __init__(self, reduce_fn: Callable = torch.mean):
		super().__init__()
		self.reduce_fn = reduce_fn

	def compute_score(self, input_: Tensor, target: Tensor) -> Tensor:
		score = (input_ - target) ** 2
		score = self.reduce_fn(score)
		return score
