
from abc import ABC
from torch.nn import Module
from typing import Callable, Generic, Iterable, Optional, TypeVar

Input = TypeVar("Input")
Target = TypeVar("Target")
Output = TypeVar("Output")
T = TypeVar("T")
U = TypeVar("U")


class Metric(Module, Callable, ABC, Generic[Input, Target, Output]):
	"""
		Base class for metric modules.

		Abstract methods:
			- compute_score(self, input_: Input, target: Target) -> Output:
	"""
	def forward(self, input_: Input, target: Target) -> Output:
		return self.compute_score(input_, target)

	def compute_score(self, input_: Input, target: Target) -> Output:
		raise NotImplemented("Abstract method")


class IncrementalMetric(Module, Callable, ABC, Generic[T, U]):
	"""
		Base class for incremental metrics modules, which wrap a metric and compute a continue value on the scores.

		Abstract methods:
			- reset(self):
			- add(self, value: T):
			- get_current(self) -> Optional[U]:
			- is_empty(self) -> bool:
	"""
	def reset(self):
		"""
			Reset the current incremental value.
		"""
		raise NotImplemented("Abstract method")

	def add(self, value: T):
		"""
			Add a value to the incremental score.

			:param value: The value to add to the current incremental metric value.
		"""
		raise NotImplemented("Abstract method")

	def is_empty(self) -> bool:
		"""
			:return: Return True if no value has been added to the incremental score.
		"""
		raise NotImplemented("Abstract method")

	def get_current(self) -> Optional[U]:
		"""
			Get the current incremental score.

			:return: The current incremental metric value.
		"""
		raise NotImplemented("Abstract method")

	def add_values(self, values: Iterable[T]):
		"""
			Add a list of scores to the current incremental value.

			:param values: Add a of values to incremental metric.
		"""
		for value in values:
			self.add(value)

	def forward(self, value: T) -> Optional[U]:
		"""
			:param value: Add a value to the metric and returns the current incremental value.
			:return: The current incremental metric value.
		"""
		self.add(value)
		return self.get_current()
