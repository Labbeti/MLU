
from mlu.metrics.base import Metric, Input, Target, Output
from torch import Tensor


class CIDER(Metric):
	"""
		Consensus-based Image Description Evaluation
	"""
	def __init__(self, ngrams_max: int):
		super().__init__()
		self.ngrams_max = ngrams_max

	def compute_score(self, input_: Input, target: Target) -> Output:
		raise NotImplementedError
