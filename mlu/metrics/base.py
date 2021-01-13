
from abc import ABC
from torch import Tensor
from torch.nn import Module
from typing import Callable, Generic, Iterable, List, Optional, TypeVar

T_Input = TypeVar("T_Input")
T_Target = TypeVar("T_Target")
T_Output = TypeVar("T_Output")


class Metric(Module, Callable, ABC, Generic[T_Input, T_Target, T_Output]):
	"""
		Base class for metric modules.

		Abstract methods:
			- forward_impl_(self, input_: T_Input, target: T_Target) -> T_Output:
	"""
	def forward(self, input_: T_Input, target: T_Target) -> T_Output:
		return self.compute_score(input_, target)

	def compute_score(self, input_: T_Input, target: T_Target) -> T_Output:
		raise NotImplementedError("Abstract method")


class IncrementalMetric(Module, Callable, ABC):
	"""
		Base class for incremental metrics modules, which wrap a metric and compute a continue value on the scores.

		Abstract methods:
			- reset(self):
			- add(self, value: Tensor):
			- get_current(self) -> Optional[Tensor]:
			- is_empty(self) -> bool:
	"""
	def reset(self):
		"""
			Reset the current incremental value.
		"""
		raise NotImplementedError("Abstract method")

	def add(self, value: Tensor):
		"""
			Add a value to the incremental score.

			:param value: The value to add to the current incremental metric value.
		"""
		raise NotImplementedError("Abstract method")

	def is_empty(self) -> bool:
		"""
			:return: Return True if no value has been added to the incremental score.
		"""
		raise NotImplementedError("Abstract method")

	def get_current(self) -> Optional[Tensor]:
		"""
			Get the current incremental score.

			:return: The current incremental metric value.
		"""
		raise NotImplementedError("Abstract method")

	def add_list(self, values: Iterable[Tensor]):
		"""
			Add a list of scores to the current incremental value.

			:param values: Add a of values to incremental metric.
		"""
		for value in values:
			self.add(value)

	def forward(self, value: Tensor) -> Optional[Tensor]:
		"""
			:param value: Add a value to the metric and returns the current incremental value.
		"""
		self.add(value)
		return self.get_current()
