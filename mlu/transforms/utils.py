
from mlu.transforms.base import Transform
from torch import Tensor


class Identity(Transform):
	def __init__(self, p: float = 1.0):
		"""
			Identity transform.

			:param p: The probability to apply the transform. (default: 1.0)
		"""
		super().__init__(p=p)

	def process(self, x: Tensor) -> Tensor:
		return x

	def is_image_transform(self) -> bool:
		return True

	def is_waveform_transform(self) -> bool:
		return True

	def is_spectrogram_transform(self) -> bool:
		return True


def default_extra_repr(obj: object, skip_private_attr: bool = True) -> str:
	attributes = [
		f"name={value}" for name, value in obj.__dict__.items() if not skip_private_attr or not name.startswith("_")
	]
	return ", ".join(attributes)
