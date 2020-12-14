
from mlu.metrics.base import Metric
from torch import Tensor
from typing import Callable


class MetricWrapper(Metric):
	def __init__(self, callable: Callable):
		super().__init__()
		self.callable = callable

	def compute_score(self, input_: Tensor, target: Tensor) -> Tensor:
		return self.callable(input_, target)
