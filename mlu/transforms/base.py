
from abc import ABC
from random import random
from torch import Tensor
from torch.nn import Module
from typing import Callable, Generic, TypeVar


T_InputType = TypeVar("T_InputType")
T_OutputType = TypeVar("T_OutputType")


class Transform(Module, Callable, ABC, Generic[T_InputType, T_OutputType]):
	def __init__(self, p: float = 1.0):
		super().__init__()
		self.p = p

	def forward(self, x: Tensor) -> Tensor:
		if self.p < 1.0 and random() <= self.p:
			return self.apply(x)
		else:
			return x

	def set_probability(self, p: float):
		self.p = p

	def get_probability(self) -> float:
		return self.p

	def is_image_transform(self) -> bool:
		return False

	def is_waveform_transform(self) -> bool:
		return False

	def is_spectrogram_transform(self) -> bool:
		return False

	def apply(self, x: T_InputType) -> T_OutputType:
		raise NotImplementedError("Abstract method")


class ImageTransform(Transform, ABC, Generic[T_InputType, T_OutputType]):
	def __init__(self, p: float = 1.0):
		super().__init__(p)

	def is_image_transform(self) -> bool:
		return True


class WaveformTransform(Transform, ABC, Generic[T_InputType, T_OutputType]):
	def __init__(self, p: float = 1.0):
		super().__init__(p)

	def is_waveform_transform(self) -> bool:
		return True


class SpectrogramTransform(Transform, ABC, Generic[T_InputType, T_OutputType]):
	def __init__(self, p: float = 1.0):
		super().__init__(p)

	def is_spectrogram_transform(self) -> bool:
		return True
