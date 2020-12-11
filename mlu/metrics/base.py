
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
		raise NotImplementedError("Abstract method")

	def add(self, item: Tensor):
		raise NotImplementedError("Abstract method")

	def get_current(self) -> Optional[Tensor]:
		raise NotImplementedError("Abstract method")
