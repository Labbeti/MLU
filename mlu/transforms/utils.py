
import random

from mlu.transforms.base import Transform

from torch import Tensor
from typing import Any, Callable, Optional, Sequence


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


class Compose(Transform):
	def __init__(self, *transforms: Transform, p: float = 1.0):
		"""
			Compose a list of transforms for apply them sequentially.

			:param transforms: The list of transforms to apply.
			:param p: The probability to apply the transform.
		"""
		super().__init__(p=p)
		self.transforms = list(transforms)

	def apply(self, x: Any) -> Any:
		for transform in self.transforms:
			x = transform(x)
		return x

	def is_image_transform(self) -> bool:
		return self._is_transform_type(lambda t: t.is_image_transform())

	def is_waveform_transform(self) -> bool:
		return self._is_transform_type(lambda t: t.is_waveform_transform())

	def is_spectrogram_transform(self) -> bool:
		return self._is_transform_type(lambda t: t.is_spectrogram_transform())

	def _is_transform_type(self, is_type_fn: Callable[[Transform], bool]) -> bool:
		return len(self.transforms) > 0 and isinstance(self.transforms[0], Transform) and is_type_fn(self.transforms[0])


class RandomChoice(Transform):
	def __init__(
		self,
		*transforms: Transform,
		nb_choices: int = 1,
		weights: Optional[Sequence[float]] = None,
		p: float = 1.0,
	):
		"""
			Select randomly k transforms in a list and apply them sequentially.

			An augmentation can be chosen multiple times if nb_choices > 1.

			:param transforms: The list of transforms from we choose the apply a transform.
			:param nb_choices: The number of transforms to choose.
			:param weights: The probabilities to choose the transform.
			:param p: The probability to apply the transform.
		"""
		super().__init__(p=p)
		self.transforms = list(transforms)
		self.nb_choices = nb_choices
		self.weights = weights

	def apply(self, x: Any) -> Any:
		transforms = random.choices(self.transforms, weights=self.weights, k=self.nb_choices)
		for transform in transforms:
			x = transform(x)
		return x

	def is_image_transform(self) -> bool:
		return self._is_transform_type(lambda t: t.is_image_transform())

	def is_waveform_transform(self) -> bool:
		return self._is_transform_type(lambda t: t.is_waveform_transform())

	def is_spectrogram_transform(self) -> bool:
		return self._is_transform_type(lambda t: t.is_spectrogram_transform())

	def _is_transform_type(self, is_type_fn: Callable[[Transform], bool]) -> bool:
		return all([isinstance(transform, Transform) and is_type_fn(transform) for transform in self.transforms])


class Permute(Transform):
	def __init__(self, *dims, p: float = 1.0):
		super().__init__(p=p)
		self.dims = list(dims)

	def apply(self, x: Tensor) -> Tensor:
		out = x.permute(*self.dims)
		return out
