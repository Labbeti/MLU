
from mlu.transforms.base import ImageTransform, Transform
from mlu.transforms.converters import ToPIL, ToTensor

from PIL import Image

from torch import Tensor
from typing import Any, Callable, Optional


class ProcessTransformWrap(Transform):
	def __init__(
		self,
		transform: Optional[Transform],
		pre_convert: Optional[Callable],
		post_convert: Optional[Callable],
		p: float = 1.0,
	):
		"""
			:param transform:
			:param pre_convert:
			:param post_convert:
			:param p: The probability to apply the transform.
		"""
		super().__init__(p=p)
		self.transform = transform
		self.pre_convert = pre_convert
		self.post_convert = post_convert
		self._update_callables()

	def apply(self, x: Any) -> Any:
		for callable_ in self._callables:
			x = callable_(x)
		return x

	def is_image_transform(self) -> bool:
		return self.transform.is_image_transform()

	def is_spectrogram_transform(self) -> bool:
		return self.transform.is_spectrogram_transform()

	def is_waveform_transform(self) -> bool:
		return self.transform.is_waveform_transform()

	def unwrap(self) -> Optional[Transform]:
		return self.transform

	def _update_callables(self):
		self._callables = []

		if self.pre_convert is not None:
			self._callables.append(self.pre_convert)
		if self.transform is not None:
			self._callables.append(self.transform)
		if self.post_convert is not None:
			self._callables.append(self.post_convert)


class PILInternalWrap(ProcessTransformWrap):
	def __init__(self, pil_transform: ImageTransform, mode: Optional[str] = "RGB", p: float = 1.0):
		"""
			Class that convert tensor to PIL image internally for apply PIL transforms.
			Tensors images must have the shape (width, height, 3).

			:param pil_transform:
			:param mode:
			:param p: The probability to apply the transform.
		"""
		super().__init__(
			transform=pil_transform,
			pre_convert=ToPIL(mode=mode),
			post_convert=ToTensor(),
			p=p,
		)

	def apply(self, x: Tensor) -> Tensor:
		return super().apply(x)


class TensorInternalWrap(ProcessTransformWrap):
	def __init__(self, pil_transform: ImageTransform, p: float = 1.0):
		"""
			Class that convert PIL image to tensor internally for apply tensor transforms.
			Tensors images will have the shape (width, height, 3).

			:param pil_transform:
			:param p: The probability to apply the transform.
		"""
		super().__init__(
			transform=pil_transform,
			pre_convert=ToTensor(),
			post_convert=ToPIL(mode=None),
			p=p,
		)

	def apply(self, x: Image.Image) -> Image.Image:
		self.post_convert.mode = x.mode
		return self.post_convert(self.transform(self.pre_convert(x)))


class TransformWrap(Transform):
	def __init__(
		self,
		callable_: Callable,
		image_transform: bool = False,
		waveform_transform: bool = False,
		spectrogram_transform: bool = False,
		p: float = 1.0,
	):
		"""
			Wrap a callable object to Transform.

			:param callable_:
			:param image_transform:
			:param waveform_transform:
			:param spectrogram_transform:
			:param p: The probability to apply the transform.
		"""
		super().__init__(p=p)
		self.callable_ = callable_
		self.image_transform = image_transform
		self.waveform_transform = waveform_transform
		self.spectrogram_transform = spectrogram_transform

	def apply(self, x: Tensor) -> Tensor:
		return self.callable_(x)

	def is_image_transform(self) -> bool:
		return self.image_transform

	def is_waveform_transform(self) -> bool:
		return self.waveform_transform

	def is_spectrogram_transform(self) -> bool:
		return self.spectrogram_transform

	def unwrap(self) -> Callable:
		return self.callable_

	def extra_repr(self) -> str:
		if hasattr(self.callable_, "__class__"):
			return self.callable_.__class__.__name__
		else:
			return ""
