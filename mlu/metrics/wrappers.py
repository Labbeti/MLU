
from mlu.metrics.base import Metric, IncrementalMetric, Input, Target, Output, T, U
from mlu.metrics.incremental import IncrementalMean

from torch.nn import Module
from typing import Callable, Dict, List, Optional


class MetricDict(Dict[str, Metric], Metric):
	def __init__(self, *args, prefix: Optional[str] = None, **kwargs):
		dict.__init__(self, *args, **kwargs)
		Metric.__init__(self)
		self.prefix = prefix

	def compute_score(self, input_: Input, target: Target) -> Dict[str, Output]:
		return {f"{self.prefix}{metric_name}": metric(input_, target) for metric_name, metric in self.items()}

	def __hash__(self) -> int:
		return hash(tuple(sorted(self.items())))


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
			:param reduce_fn: The reduction function to apply.
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

		if isinstance(callable_, Module):
			self.add_module("callable", callable_)

	def compute_score(self, input_: Input, target: Target) -> Output:
		score = self.sub_call(input_, target)
		if self.reduce_fn is not None:
			score = self.reduce_fn(score)
		return score


class IncrementalWrapper(Metric):
	def __init__(
		self,
		metric: Metric,
		incremental_metric: IncrementalMetric = IncrementalMean()
	):
		"""
			Compute an incremental score (mean or std) of a metric.

			:param metric: The metric used to compute each score.
			:param incremental_metric: The incremental (continue) way to compute the mean or std.
		"""
		super().__init__()
		self.metric = metric
		self.continue_metric = incremental_metric

	def compute_score(self, input_: Input, target: Target) -> Output:
		score = self.metric(input_, target)
		self.continue_metric.add(score)
		return self.continue_metric.get_current()


class IncrementalListWrapper(Metric):
	def __init__(self, metric: Metric, incremental_metric_list: Optional[List[IncrementalMetric]] = None):
		"""
			Compute a list of incremental scores (mean or std) of a metric.

			:param metric: The metric used to compute each score.
			:param incremental_metric_list: The list of incremental (continue) metrics for compute the mean or std.
		"""
		super().__init__()
		self.metric = metric
		self.continue_metric_list = incremental_metric_list if incremental_metric_list is not None else []

	def compute_score(self, input_: Input, target: Target) -> List[Output]:
		score = self.metric(input_, target)
		for continue_metric in self.continue_metric_list:
			continue_metric.add(score)
		return [continue_metric.get_current() for continue_metric in self.continue_metric_list]


class MetricDictPrePostFix(Dict[str, Metric], Metric):
	def __init__(
		self,
		default_prefix: Optional[str] = None,
		default_suffix: Optional[str] = None,
		**kwargs,
	):
		super().__init__(**kwargs)
		self.default_prefix = default_prefix
		self.default_suffix = default_suffix

		if self.default_prefix is not None or self.default_suffix is not None:
			for metric_name, metric in self.items():
				self.add_metric(metric, metric_name)

	def compute_score(self, input_: Input, target: Target) -> Dict[str, Output]:
		return {metric_name: metric(input_, target) for metric_name, metric in self.items()}

	def add_metric(
		self,
		metric: Metric,
		name: Optional[str] = None,
	):
		"""
			Add a metric to the MetricDict.

			:param metric: The metric to add.
			:param name: The name of the metric.
				If None, the name of metric class will be used. (default: None)
		"""
		if name is None:
			name = metric.__name__

		if "/" in name or self.default_prefix is None:
			complete_name = name
		else:
			complete_name = f"{self.default_prefix}/{name}"

		if self.default_suffix is not None:
			complete_name += self.default_suffix

		self[complete_name] = metric


class IncrementalList(IncrementalMetric):
	def __init__(self, incremental_list: List[IncrementalMetric]):
		super().__init__()
		self.incremental_list = incremental_list

	def reset(self):
		for incremental in self.incremental_list:
			incremental.reset()

	def add(self, value: T):
		for incremental in self.incremental_list:
			incremental.add(value)

	def is_empty(self) -> List[bool]:
		return [
			incremental.is_empty()
			for incremental in self.incremental_list
		]

	def get_current(self) -> List[Optional[U]]:
		return [
			incremental.get_current()
			for incremental in self.incremental_list
		]
