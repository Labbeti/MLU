
from torch import Tensor
from typing import Optional
from mlu.metrics.base import Metric


class EqMetric(Metric):
	def __init__(self, dim: Optional[int] = None):
		"""
			Equal metric along a dimension.

			:param dim: The dimension to compute the score.
		"""
		super().__init__()
		self.dim = dim

	def compute_score(self, input_: Tensor, target: Tensor) -> Tensor:
		return input_.eq(target).all(self.dim).float()
