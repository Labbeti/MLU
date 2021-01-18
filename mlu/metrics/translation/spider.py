
from mlu.metrics.base import Metric, Input, Target, Output


class SPIDER(Metric):
	def compute_score(self, input_: Input, target: Target) -> Output:
		raise NotImplementedError
