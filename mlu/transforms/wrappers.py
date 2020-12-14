
from mlu.transforms.base import ImageTransform, Transform, T_InputType, T_OutputType
from mlu.transforms.convert import ToPIL, ToTensor

from PIL import Image

from torch import Tensor
from typing import Any, Callable, Optional, Generic


class ConversionWrapper(Transform, Generic[T_InputType, T_OutputType]):
	def __init__(self, transform: ImageTransform, pre_convert: Callable, post_convert: Callable, p: float = 1.0):
		super().__init__(p=p)
		self.transform = transform
		self.pre_convert = pre_convert
		self.post_convert = post_convert

	def apply(self, x: Any) -> Any:
		return self.post_convert(self.transform(self.pre_convert(x)))

	def is_image_transform(self) -> bool:
		return self.transform.is_image_transform()

	def is_spectrogram_transform(self) -> bool:
		return self.transform.is_spectrogram_transform()

	def is_waveform_transform(self) -> bool:
		return self.transform.is_waveform_transform()


class PILInternalWrapper(ConversionWrapper[Tensor, Tensor]):
	"""
		Class that convert tensor to PIL image internally for apply PIL transforms.
		Tensors images must have the shape (width, height, 3).
	"""
	def __init__(self, pil_transform: ImageTransform, mode: Optional[str] = "RGB", p: float = 1.0):
		super().__init__(
			transform=pil_transform,
			pre_convert=ToPIL(mode=mode),
			post_convert=ToTensor(),
			p=p,
		)

	def apply(self, x: Tensor) -> Tensor:
		return super().apply(x)


class TensorInternalWrapper(ConversionWrapper[Image.Image, Image.Image]):
	"""
		Class that convert PIL image to tensor internally for apply tensor transforms.
		Tensors images will have the shape (width, height, 3).
	"""
	def __init__(self, pil_transform: ImageTransform, p: float = 1.0):
		super().__init__(
			transform=pil_transform,
			pre_convert=ToTensor(),
			post_convert=ToPIL(mode=None),
			p=p,
		)

	def apply(self, x: Image.Image) -> Image.Image:
		self.post_convert.mode = x.mode
		return self.post_convert(self.transform(self.pre_convert(x)))


class TransformWrapper(Transform):
	def __init__(
		self,
		callable: Callable,
		image_transform: bool = False,
		waveform_transform: bool = False,
		spectrogram_transform: bool = False,
		p: float = 1.0
	):
		"""
			Wrap a callable object to Transform.

			:param p: The probability to apply the transform.
		"""
		super().__init__(p=p)
		self.callable = callable
		self.image_transform = image_transform
		self.waveform_transform = waveform_transform
		self.spectrogram_transform = spectrogram_transform

	def apply(self, x: Tensor) -> Tensor:
		return self.callable(x)

	def is_image_transform(self) -> bool:
		return self.image_transform

	def is_waveform_transform(self) -> bool:
		return self.waveform_transform

	def is_spectrogram_transform(self) -> bool:
		return self.spectrogram_transform
