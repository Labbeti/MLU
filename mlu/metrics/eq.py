
from mlu.metrics.base import Metric
from torch import Tensor


class EqMetric(Metric):
	def __init__(self, dim: int):
		super().__init__()
		self.dim = dim

	def compute_score(self, input_: Tensor, target: Tensor) -> Tensor:
		return input_.eq(target).all(self.dim).float()
