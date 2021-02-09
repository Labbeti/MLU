
from mlu.metrics.base import Metric
from torch import Tensor


class ClassRatio(Metric):
	def compute_score(self, input_: Tensor, target: Tensor) -> Tensor:
		summed = target.sum(dim=0)
		summed = summed[summed > 0]
		ratio = summed.min() / summed.max()
		return ratio
