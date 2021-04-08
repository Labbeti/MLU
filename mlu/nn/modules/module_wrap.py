
from torch.nn import Module
from typing import Callable


class ModuleWrap(Module):
	"""
		Wrapper for a Callable object with a forward() method.
	"""
	def __init__(self, callable_: Callable):
		super().__init__()
		self.callable_ = callable_

	def forward(self, *args, **kwargs):
		return self.callable_(*args, **kwargs)

	def extra_repr(self) -> str:
		if hasattr(self.callable_, "__name__"):
			return self.callable_.__name__
		elif hasattr(self.callable_, "__class__"):
			return self.callable_.__class__.__name__
		else:
			return ""
