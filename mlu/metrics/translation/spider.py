
from mlu.metrics.base import Metric, T_Input, T_Target, T_Output


class SPIDER(Metric):
	def compute_score(self, input_: T_Input, target: T_Target) -> T_Output:
		raise NotImplementedError
