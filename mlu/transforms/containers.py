
import random

from abc import ABC
from mlu.transforms.base import Transform
from mlu.transforms.wrappers import TransformWrap
from torch.nn import Module
from typing import Any, Callable, List, Optional, Sequence


class Container(Transform, ABC):
	def __init__(self, *transforms: Callable, p: float = 1.0):
		super().__init__(p=p)
		self._transforms = list(transforms)

		for i, transform in enumerate(self._transforms):
			if not isinstance(transform, Module):
				transform = TransformWrap(transform)
			self.add_module(str(i), transform)

	def __getitem__(self, index: int) -> Callable:
		return self._transforms[index]

	def get_transforms(self) -> List[Callable]:
		return self._transforms

	def is_image_transform(self) -> bool:
		return all([isinstance(transform, Transform) and transform.is_image_transform() for transform in self._transforms])

	def is_waveform_transform(self) -> bool:
		return all([isinstance(transform, Transform) and transform.is_waveform_transform() for transform in self._transforms])

	def is_spectrogram_transform(self) -> bool:
		return all([isinstance(transform, Transform) and transform.is_spectrogram_transform() for transform in self._transforms])


class Compose(Container):
	def __init__(self, *transforms: Callable, p: float = 1.0):
		"""
			Compose a list of transforms for apply them sequentially.

			:param transforms: The list of transforms to apply.
			:param p: The probability to apply the transform. (default: 1.0)
		"""
		super().__init__(*transforms, p=p)

	def apply(self, x: Any) -> Any:
		for transform in self.get_transforms():
			x = transform(x)
		return x


class RandomChoice(Container):
	def __init__(
		self,
		*transforms: Callable,
		nb_choices: int = 1,
		weights: Optional[Sequence[float]] = None,
		p: float = 1.0,
	):
		"""
			Select randomly k transforms in a list and apply them sequentially.

			An transform can be chosen multiple times if nb_choices > 1. (with replacement)

			:param transforms: The list of transforms from we choose the apply a transform.
			:param nb_choices: The number of transforms to choose. (default: 1)
			:param weights: The probabilities to choose the transform. (default: None)
			:param p: The probability to apply the transform. (default: 1.0)
		"""
		super().__init__(*transforms, p=p)
		self.nb_choices = nb_choices
		self.weights = weights

	def apply(self, x: Any) -> Any:
		transforms = random.choices(self.get_transforms(), weights=self.weights, k=self.nb_choices)
		for transform in transforms:
			x = transform(x)
		return x
