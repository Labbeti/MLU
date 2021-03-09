
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
			if not callable(transform):
				raise RuntimeError(f"Cannot add non-callable object '{type(transform)}'.")
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

	def process(self, x: Any) -> Any:
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

	def process(self, x: Any) -> Any:
		transforms = random.choices(self.get_transforms(), weights=self.weights, k=self.nb_choices)
		for transform in transforms:
			x = transform(x)
		return x


class PoolRandomChoice(Container):
	def __init__(
		self,
		pool_transforms: List[Callable],
		transform_to_spec: Optional[Callable],
		is_spec_transform: Callable,
		p: float = 1.0
	):
		"""
			TODO
		"""
		super().__init__(*pool_transforms, p=p)
		self._transform_to_spec = transform_to_spec
		self._is_spec_transform = is_spec_transform
		self._transform_composed = None

		self._build()

	def process(self, x: Any) -> Any:
		return self._transform_composed(x)

	def _build(self):
		pool = self.get_transforms()
		transform_to_spec = self._transform_to_spec
		is_spec_transform = self._is_spec_transform

		pool_new = []
		for augm in pool:
			transforms = []

			if augm is not None:
				if transform_to_spec is not None:
					# Add transform to spectrogram before or after each augment depending on his internal type.
					if is_spec_transform(augm):
						transforms.append(transform_to_spec)
						transforms.append(augm)
					else:
						transforms.append(augm)
						transforms.append(transform_to_spec)
				else:
					transforms.append(augm)
			elif transform_to_spec is not None:
				transforms.append(transform_to_spec)

			if len(transforms) == 0:
				raise RuntimeError("Found an empty list of transforms.")
			elif len(transforms) == 1:
				pool_new.append(transforms[0])
			else:
				pool_new.append(Compose(*transforms))

		if len(pool_new) == 0:
			raise RuntimeError("Found an empty transform pool.")
		elif len(pool_new) == 1:
			transform_composed = pool_new[0]
		else:
			transform_composed = RandomChoice(*pool_new)

		self._transform_composed = transform_composed
