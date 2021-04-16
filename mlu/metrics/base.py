
from abc import ABC
from torch import Tensor
from torch.nn import Module
from typing import Iterable, Optional


class Metric(Module, ABC):
	"""
		Base class for metric modules.

		Abstract methods:
			- compute_score(self, input_, target):
	"""
	def __init__(self, input_to_cpu: bool = True):
		super().__init__()
		self.input_to_cpu = input_to_cpu

	def forward(self, pred, target):
		if self.input_to_cpu:
			if isinstance(pred, Tensor):
				pred = pred.cpu()
			if isinstance(target, Tensor):
				target = target.cpu()
		score = self.compute_score(pred, target)
		return score

	def compute_score(self, pred, target):
		raise NotImplemented('Abstract method')


class IncrementalMetric(Module, ABC):
	"""
		Base class for incremental metrics modules, which wrap a metric and compute a continue value on the scores.

		Abstract methods:
			- reset(self):
			- add(self, value: T):
			- get_current(self) -> Optional:
			- is_empty(self) -> bool:
	"""
	def reset(self):
		"""
			Reset the current incremental value.
		"""
		raise NotImplemented('Abstract method')

	def add(self, value):
		"""
			Add a value to the incremental score.

			:param value: The value to add to the current incremental metric value.
		"""
		raise NotImplemented('Abstract method')

	def is_empty(self) -> bool:
		"""
			:return: Return True if no value has been added to the incremental score.
		"""
		raise NotImplemented('Abstract method')

	def get_current(self) -> Optional:
		"""
			Get the current incremental score.

			:return: The current incremental metric value.
		"""
		raise NotImplemented('Abstract method')

	def add_values(self, values: Iterable):
		"""
			Add a list of scores to the current incremental value.

			:param values: Add a of values to incremental metric.
		"""
		for value in values:
			self.add(value)

	def forward(self, value) -> Optional:
		"""
			:param value: Add a value to the metric and returns the current incremental value.
			:return: The current incremental metric value.
		"""
		self.add(value)
		return self.get_current()
