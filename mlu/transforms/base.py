
from abc import ABC
from random import random
from torch.nn import Module
from typing import Callable, Generic, TypeVar


Input = TypeVar("Input")
Output = TypeVar("Output")


class Transform(Module, Callable, ABC, Generic[Input, Output]):
	def __init__(self, p: float = 1.0):
		"""
			Base class for all Transforms.

			:param p: The probability to apply the transform.
		"""
		super().__init__()
		assert 0.0 <= p <= 1.0, "Probability must be a float in range [0, 1]."
		self.p = p

	def forward(self, x: Input) -> Output:
		if self.p == 1.0 or random() <= self.p:
			return self.apply(x)
		else:
			return x

	def set_p(self, p: float):
		"""
			Set the internal probability to apply the transform.

			:param p: The new probability to apply the transform in range [0, 1].
		"""
		self.p = p

	def get_p(self) -> float:
		"""
			Get the probability to apply the transform.

			:return: The current probability to apply the transform in range [0, 1].
		"""
		return self.p

	def is_image_transform(self) -> bool:
		return False

	def is_waveform_transform(self) -> bool:
		return False

	def is_spectrogram_transform(self) -> bool:
		return False

	def apply(self, x: Input) -> Output:
		raise NotImplementedError("Abstract method")


class ImageTransform(Transform, ABC, Generic[Input, Output]):
	def __init__(self, p: float = 1.0):
		super().__init__(p=p)

	def is_image_transform(self) -> bool:
		return True


class WaveformTransform(Transform, ABC, Generic[Input, Output]):
	def __init__(self, p: float = 1.0):
		super().__init__(p=p)

	def is_waveform_transform(self) -> bool:
		return True


class SpectrogramTransform(Transform, ABC, Generic[Input, Output]):
	def __init__(self, p: float = 1.0):
		super().__init__(p=p)

	def is_spectrogram_transform(self) -> bool:
		return True
