
from mlu.metrics.base import Metric, IncrementalMetric, Input, Target, Output, T, U
from mlu.metrics.incremental import IncrementalMean

from torch import Tensor
from torch.nn import Module
from typing import Callable, Dict, List, Optional


class MetricDict(Dict[str, Module], Metric):
	def __init__(self, *args, prefix: str = "", suffix: str = "", **kwargs):
		"""
			Compute score of each metric stored when forward() is called.
			Subclass of Dict[str, Metric] and Metric.

			Example :

			>>> import torch
			>>> from mlu.metrics import CategoricalAccuracy, FScore, MetricDict
			>>> input_, target = torch.rand(5, 10), torch.rand(5, 10)
			>>> metric_dict = MetricDict(acc=CategoricalAccuracy(), f1=FScore())
			>>> metric_dict(input_, target)
			... {"acc": 0.4, "f1": 0.1}

		"""
		dict.__init__(self, *args, **kwargs)
		Metric.__init__(self)
		self.prefix = prefix
		self.suffix = suffix

	def compute_score(
		self,
		input_: Input,
		target: Target,
	) -> Dict[str, Output]:
		"""
			Compute the score of each metric stored and return the dictionary of {metric_name: metric_score, ...}.
		"""
		return {
			(self.prefix + metric_name + self.suffix): metric(input_, target)
			for metric_name, metric in self.items()
		}

	def __hash__(self) -> int:
		return hash(tuple(sorted(self.items()))) + hash(self.prefix) + hash(self.suffix)

	def to_dict(self, with_pre_and_suf: bool = True) -> Dict[str, Metric]:
		"""
			:param with_pre_and_suf: If True, append the prefix and suffix to the keys metrics names. (default: True)
			:return: Return the metrics names and metrics as python dict object.
		"""
		if with_pre_and_suf:
			dic = {f"{self.prefix}{metric_name}{self.suffix}": metric for metric_name, metric in self.items()}
		else:
			dic = dict(self)
		return dic


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
		self.callable_ = callable_
		self.reduce_fn = reduce_fn

		if use_input and use_target:
			self.sub_call = self._sub_call_both
		elif use_input:
			self.sub_call = self._sub_call_input
		elif use_target:
			self.sub_call = self._sub_call_target
		else:
			self.sub_call = self._sub_call_none

	def compute_score(self, input_: Input, target: Target) -> Output:
		score = self.sub_call(input_, target)
		if self.reduce_fn is not None:
			score = self.reduce_fn(score)
		return score

	def _sub_call_both(self, input_: Tensor, target: Tensor) -> Tensor:
		return self.callable_(input_, target)

	def _sub_call_input(self, input_: Tensor, target: Tensor) -> Tensor:
		return self.callable_(input_)

	def _sub_call_target(self, input_: Tensor, target: Tensor) -> Tensor:
		return self.callable_(target)

	def _sub_call_none(self, input_: Tensor, target: Tensor) -> Tensor:
		return self.callable_()


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
