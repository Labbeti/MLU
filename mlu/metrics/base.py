
from abc import ABC
from torch import Tensor
from torch.nn import Module
from typing import Callable, Optional


class Metric(Module, Callable, ABC):
	"""
		Base class for metric modules.
	"""
	def forward(self, input_: Tensor, target: Tensor) -> Tensor:
		return self.compute_score(input_, target)

	def compute_score(self, input_: Tensor, target: Tensor) -> Tensor:
		raise NotImplementedError("Abstract method")


class IncrementalMetric(Module, Callable, ABC):
	"""
		Base class for continue metrics modules, which wrap a metric and compute a continue value on the scores.
	"""
	def reset(self):
		"""
			Reset the current incremental value.
		"""
		raise NotImplementedError("Abstract method")

	def add(self, value: Tensor):
		"""
			Add a value to the incremental score.
		"""
		raise NotImplementedError("Abstract method")

	def get_current(self) -> Optional[Tensor]:
		"""
			Get the current incremental score.
		"""
		raise NotImplementedError("Abstract method")

	def is_empty(self) -> bool:
		"""
			Return True if no value has been added to the incremental score.
		"""
		raise NotImplementedError("Abstract method")
