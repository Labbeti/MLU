
from mlu.metrics.base import Metric, IncrementalMetric, T_Input, T_Target, T_Output
from mlu.metrics.incremental import IncrementalMean
from typing import Callable, List, Optional


class MetricWrapper(Metric):
	def __init__(
		self,
		callable_: Callable,
		use_input: bool = True,
		use_target: bool = True,
		reduce_fn: Optional[Callable] = None,
	):
		"""
			Wrapper of a callable function or class for comply with Metric typing.

			:param callable_: The callable object to wrap.
			:param use_input: If True, the input_ argument will be passed as argument to the callable object wrapped.
			:param use_target: If True, the target argument will be passed as argument to the callable object wrapped.
		"""
		super().__init__()
		self.reduce_fn = reduce_fn

		if use_input and use_target:
			self.sub_call = lambda input_, target: callable_(input_, target)
		elif use_input:
			self.sub_call = lambda input_, target: callable_(input_)
		elif use_target:
			self.sub_call = lambda input_, target: callable_(target)
		else:
			self.sub_call = lambda input_, target: callable_()

	def compute_score(self, input_: T_Input, target: T_Target) -> T_Output:
		score = self.sub_call(input_, target)
		if self.reduce_fn is not None:
			score = self.reduce_fn(score)
		return score


class IncrementalWrapper(Metric):
	def __init__(
		self,
		metric: Metric,
		continue_metric: IncrementalMetric = IncrementalMean()
	):
		"""
			Compute an incremental score (mean or std) of a metric.

			:param metric: The metric used to compute each score.
			:param continue_metric: The incremental (continue) way to compute the mean or std.
		"""
		super().__init__()
		self.metric = metric
		self.continue_metric = continue_metric

	def compute_score(self, input_: T_Input, target: T_Target) -> T_Output:
		score = self.metric(input_, target)
		self.continue_metric.add(score)
		return self.continue_metric.get_current()


class IncrementalListWrapper(Metric):
	def __init__(self, metric: Metric, continue_metric_list: Optional[List[IncrementalMetric]] = None):
		"""
			Compute a list of incremental scores (mean or std) of a metric.

			:param metric: The metric used to compute each score.
			:param continue_metric_list: The list of incremental (continue) metrics for compute the mean or std.
		"""
		super().__init__()
		self.metric = metric
		self.continue_metric_list = continue_metric_list if continue_metric_list is not None else []

	def compute_score(self, input_: T_Input, target: T_Target) -> List[T_Output]:
		score = self.metric(input_, target)
		for continue_metric in self.continue_metric_list:
			continue_metric.add(score)
		return [continue_metric.get_current() for continue_metric in self.continue_metric_list]
