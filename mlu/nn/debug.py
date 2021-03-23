
from torch.nn import Module
from typing import Callable


class Print(Module):
	def __init__(self, format_: Callable = lambda x: ""):
		super().__init__()
		self.format_ = format_

	def forward(self, x):
		print(self.format_(x))
		return x


class Assert(Module):
	def __init__(self, assertion: Callable, msg: str = "", msg_fn: Callable = lambda x: ""):
		super().__init__()
		self.assertion = assertion
		self.msg = msg
		self.msg_fn = msg_fn

	def forward(self, x):
		assert self.assertion(x), str(self.msg) + str(self.msg_fn(x))
		return x
