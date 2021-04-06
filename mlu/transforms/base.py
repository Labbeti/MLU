
import random

from abc import ABC
from torch.nn import Module
from typing import Callable, Generic, TypeVar


Input = TypeVar("Input")
Output = TypeVar("Output")


class Transform(Module, Callable, ABC, Generic[Input, Output]):
	def __init__(self, p: float = 1.0):
		"""
			Base class for all Transforms.

			:param p: The probability to apply the transform. (default: 1.0)
		"""
		if not isinstance(p, float) or not(0.0 <= p <= 1.0):
			raise ValueError("Probability parameter 'p' must be a float in range [0, 1].")

		super().__init__()
		self.p = p

	def forward(self, *x: Input) -> Output:
		if self.p >= 1.0 or random.random() <= self.p:
			return self.process(*x)
		else:
			return x if len(x) > 1 else x[0]

	def is_image_transform(self) -> bool:
		"""
			:return: True if the transform must be applied to images.
		"""
		return False

	def is_waveform_transform(self) -> bool:
		"""
			:return: True if the transform must be applied to audio waveforms.
		"""
		return False

	def is_spectrogram_transform(self) -> bool:
		"""
			:return: True if the transform must be applied to audio spectrogram.
		"""
		return False

	def process(self, x: Input) -> Output:
		raise NotImplemented("Abstract method")


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
