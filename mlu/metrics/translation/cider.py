
from mlu.metrics.base import Metric, T_Input, T_Target, T_Output
from torch import Tensor


class CIDER(Metric):
	"""
		Consensus-based Image Description Evaluation
	"""
	def __init__(self, ngrams_max: int):
		super().__init__()
		self.ngrams_max = ngrams_max

	def compute_score(self, input_: T_Input, target: T_Target) -> T_Output:
		raise NotImplementedError
