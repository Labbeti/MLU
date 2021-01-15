
from mlu.transforms.base import Transform
from torch import Tensor


class Identity(Transform):
	def __init__(self, p: float = 1.0):
		"""
			Identity transform.

			:param p: The probability to apply the transform.
		"""
		super().__init__(p=p)

	def apply(self, x: Tensor) -> Tensor:
		return x

	def is_image_transform(self) -> bool:
		return True

	def is_waveform_transform(self) -> bool:
		return True

	def is_spectrogram_transform(self) -> bool:
		return True
