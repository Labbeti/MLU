
import random

from mlu.transforms.base import ImageTransform, T_Input, T_Output
from mlu.transforms.image.pil import (
	AutoContrast,
	Brightness,
	Color,
	Contrast,
	Equalize,
	IdentityImage,
	Posterize,
	Rotation,
	Sharpness,
	ShearX,
	ShearY,
	Solarize,
	TranslateX,
	TranslateY,
)
from typing import Any, Generic, List, Optional, Tuple, Type


"""
	The 14 default augmentations used in RandAugment with the range of their parameter.
	If no parameter is available, then the range is None.
	
	- An augment is range must have a constructor with 1 parameter.
	- An augment without range must have a constructor without any parameters.
"""
RAND_AUGMENT_DEFAULT_POOL = [
	(AutoContrast, None),
	(Brightness, (-0.9, 0.9)),
	(Color, (-0.9, 0.9)),
	(Contrast, (-0.9, 0.9)),
	(Equalize, None),
	(IdentityImage, None),
	(Posterize, (0, 4)),
	(Rotation, (-30, 30)),
	(Sharpness, (-0.9, 0.9)),
	(ShearX, (-0.3, 0.3)),
	(ShearY, (-0.3, 0.3)),
	(Solarize, (0, 256)),
	(TranslateX, (-0.3, 0.3)),
	(TranslateY, (-0.3, 0.3)),
]


class RandAugment(Generic[T_Input, T_Output], ImageTransform[T_Input, T_Output]):
	def __init__(
		self,
		nb_augm_apply: int = 1,
		magnitude: float = 2.0,
		augm_pool: Optional[List[Tuple[Type[ImageTransform], Optional[Tuple[float, float]]]]] = None,
		magnitude_policy: str = "random",
		p: float = 1.0,
	):
		"""
			Unofficial pytorch implementation of RandAugment with constant magnitude.
			Original paper : https://arxiv.org/pdf/1909.13719.pdf

			:param nb_augm_apply: The number of augmentations "N" to apply on 1 image.
			:param magnitude: The magnitude "M" used in RandAugment in range [0, 1].
				If magnitude_policy == "random", this parameter is ignored.
			:param augm_pool: The list of augmentations with their optional range.
			:param magnitude_policy: The policy to apply for control magnitude of augmentations.
				Available policies are "constant" and "random".
			:param p: The probability to apply the augmentation.
		"""
		super().__init__(p)
		self._nb_augm_apply = nb_augm_apply
		self._magnitude = magnitude
		self._augm_pool = augm_pool if augm_pool is not None else RAND_AUGMENT_DEFAULT_POOL
		self._magnitude_policy = magnitude_policy

		self._augms = _build_augms(self._augm_pool, self._magnitude)

	def apply(self, x: Any) -> Any:
		if self._magnitude_policy == "random":
			new_magnitude = random.random()
			self.set_magnitude(new_magnitude)

		augms_to_apply = random.choices(self._augms, k=self._nb_augm_apply)
		for augm in augms_to_apply:
			x = augm(x)
		return x

	def set_magnitude(self, magnitude: float):
		self._magnitude = magnitude
		self._augms = _build_augms(self._augm_pool, self._magnitude)


def _build_augms(
	augm_pool: List[Tuple[Type[ImageTransform], Optional[Tuple[float, float]]]], magnitude: float
) -> List[ImageTransform]:
	augms = []
	for i, (augm_cls, range_) in enumerate(augm_pool):
		if range_ is not None:
			min_, max_ = range_
			if min_ >= 0.0 and max_ >= 0.0:
				augm_param = (max_ - min_) * magnitude + min_
			else:
				if random.random() < abs(min_) / (abs(min_) + abs(max_)):
					augm_param = min_ * magnitude
				else:
					augm_param = max_ * magnitude
			augm = augm_cls(augm_param)
		else:
			augm = augm_cls()
		augms.append(augm)
	return augms
