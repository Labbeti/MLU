
from torch.nn import Module
from typing import Callable


class Print(Module):
	def __init__(self, *args):
		super().__init__()
		self.args = args

	def forward(self, *args):
		print(*self.args)
		return args


class Assert(Module):
	def __init__(self, assertion: Callable, msg: str):
		super().__init__()
		self.assertion = assertion
		self.msg = msg

	def forward(self, *args):
		assert self.assertion(*args), self.msg
		return args
