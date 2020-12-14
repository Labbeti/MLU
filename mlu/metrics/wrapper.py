
from mlu.metrics.base import Metric, IncrementalMetric, T_Input, T_Target, T_Output
from mlu.metrics.incremental import IncrementalMean
from torch import Tensor
from typing import Callable, List, Optional


class MetricWrapper(Metric[Tensor, Tensor, Tensor]):
	def __init__(self, callable_: Callable[[Tensor, Tensor], Tensor]):
		super().__init__()
		self.callable_ = callable_

	def compute_score(self, input_: Tensor, target: Tensor) -> Tensor:
		return self.callable_(input_, target)


class IncrementalWrapper(Metric):
	"""
		Compute an incremental score (mean or std) of a metric.
	"""
	def __init__(self, metric: Metric, continue_metric: IncrementalMetric = IncrementalMean()):
		super().__init__()
		self.metric = metric
		self.continue_metric = continue_metric

	def compute_score(self, input_: T_Input, target: T_Target) -> T_Output:
		score = self.metric(input_, target)
		self.continue_metric.add(score)
		return self.continue_metric.get_current()


class IncrementalListWrapper(Metric):
	"""
		Compute a list of incremental scores (mean or std) of a metric.
	"""
	def __init__(self, metric: Metric, continue_metric_list: Optional[List[IncrementalMetric]] = None):
		super().__init__()
		self.metric = metric
		self.continue_metric_list = continue_metric_list if continue_metric_list is not None else []

	def compute_score(self, input_: T_Input, target: T_Target) -> List[T_Output]:
		score = self.metric(input_, target)
		for continue_metric in self.continue_metric_list:
			continue_metric.add(score)
		return [continue_metric.get_current() for continue_metric in self.continue_metric_list]
