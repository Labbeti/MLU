
from mlu.metrics.base import Metric
from torch import Tensor


class DPrime(Metric):
	"""
		dPrime metric
	"""
	def compute_score(self, input_: Tensor, target: Tensor) -> Tensor:
		raise NotImplementedError
