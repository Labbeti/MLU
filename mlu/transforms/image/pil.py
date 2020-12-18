
import numpy as np

from mlu.transforms.base import ImageTransform
from mlu.utils.misc import random_rect
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
from torchvision.transforms import RandomHorizontalFlip, RandomVerticalFlip
from typing import Tuple, Union

ImageEnhance_t = "ImageEnhance"


class _Enhance(ImageTransform[Image.Image, Image.Image]):
	def __init__(self, method: ImageEnhance_t, levels: Union[float, Tuple[float, float]], p: float = 1.0):
		"""
			Enhance a PIL image with a specific method.

			:param method: The enhance method to use.
			:param levels: A constant level or a range of levels. Values should be in [-1, 1]
			:param p: The probability to apply the augmentation.
		"""
		super().__init__(p=p)
		self.method = method
		self.levels = levels if isinstance(levels, tuple) else (levels, levels)

	def apply(self, data: Image.Image) -> Image.Image:
		level = np.random.uniform(*self.levels)
		# Old enhance : enhance(0.1 + 1.9 * level).
		# Note : enhance(1) => return the same data.
		# 	Levels are in now in range [-1, 1], so we add +1 for having the same image if level = 0.
		return self.method(data).enhance(level + 1.0)


class _Blend(ImageTransform[Image.Image, Image.Image]):
	def __init__(self, augment: ImageTransform[Image.Image, Image.Image], levels: Union[float, Tuple[float, float]], p: float = 1.0):
		super().__init__(p=p)
		self.augment = augment
		self.levels = levels if isinstance(levels, tuple) else (levels, levels)

	def apply(self, data: Image.Image) -> Image.Image:
		level = np.random.uniform(*self.levels)
		return Image.blend(data, self.augment.apply(data), level)


class AutoContrast(ImageTransform[Image.Image, Image.Image]):
	def apply(self, data: Image.Image) -> Image.Image:
		return ImageOps.autocontrast(data)


class Blur(ImageTransform[Image.Image, Image.Image]):
	def apply(self, data: Image.Image) -> Image.Image:
		return data.filter(ImageFilter.BLUR)


class Brightness(ImageTransform[Image.Image, Image.Image]):
	def __init__(self, levels: Union[float, Tuple[float, float]] = 0.5, p: float = 1.0):
		"""
			Increase the brightness of an image.
			Levels must be in range [-1, 1]. -1 means a full black image, 1 means a brighter image and 0 does not change the image.
		"""
		super().__init__(p=p)
		self.enhance = _Enhance(method=ImageEnhance.Brightness, levels=levels, p=1.0)

	def apply(self, data: Image.Image) -> Image.Image:
		return self.enhance.apply(data)


class Color(ImageTransform[Image.Image, Image.Image]):
	def __init__(self, levels: Union[float, Tuple[float, float]] = 0.5, p: float = 1.0):
		"""
			Colorize an image.
			Levels must be in range [-1, 1]. -1 means a image without colors, 1 means a colorized image and 0 does not change the image.
		"""
		super().__init__(p=p)
		self.enhance = _Enhance(method=ImageEnhance.Color, levels=levels, p=1.0)

	def apply(self, data: Image.Image) -> Image.Image:
		return self.enhance.apply(data)


class Contrast(ImageTransform[Image.Image, Image.Image]):
	def __init__(self, levels: Union[float, Tuple[float, float]] = 0.5, p: float = 1.0):
		"""
			Increase contrast of an image.
			Levels must be in range [-1, 1]. -1 means a full grey image, 1 means a contrasted image and 0 does not change the image.
		"""
		super().__init__(p=p)
		self.enhance = _Enhance(method=ImageEnhance.Contrast, levels=levels, p=1.0)

	def apply(self, data: Image.Image) -> Image.Image:
		return self.enhance.apply(data)


class CutOutImgPIL(ImageTransform[Image.Image, Image.Image]):
	def __init__(
		self,
		scales: Union[float, Tuple[float, float]] = 0.5,
		fill_value: Union[float, int] = 0,
		p: float = 1.0,
	):
		"""
			Put black value in an area randomly placed.
		"""
		super().__init__(p=p)
		scales = scales if isinstance(scales, tuple) else (scales, scales)
		self.width_scale_range = scales
		self.height_scale_range = scales
		self.fill_value = fill_value

	def apply(self, x: Image.Image) -> Image.Image:
		width, height = x.width, x.height
		left, right, top, down = random_rect(width, height, self.width_scale_range, self.height_scale_range)

		output = x.copy()
		pixels = output.load()
		for i in range(left, right):
			for j in range(top, down):
				pixels[i, j] = (self.fill_value, self.fill_value, self.fill_value)
		return output


class Equalize(ImageTransform[Image.Image, Image.Image]):
	def apply(self, data: Image.Image) -> Image.Image:
		return ImageOps.equalize(data)


class HorizontalFlip(ImageTransform[Image.Image, Image.Image]):
	def __init__(self, p: float = 1.0):
		super().__init__(p=p)
		self.flip_h = RandomHorizontalFlip(1.0)

	def apply(self, data: Image.Image) -> Image.Image:
		return self.flip_h(data)


class Invert(ImageTransform[Image.Image, Image.Image]):
	def apply(self, data: Image.Image) -> Image.Image:
		return ImageOps.invert(data)


class IdentityImage(ImageTransform[Image.Image, Image.Image]):
	def apply(self, data: Image.Image) -> Image.Image:
		return data


class Posterize(ImageTransform[Image.Image, Image.Image]):
	def __init__(self, nbs_bits: Union[int, Tuple[int, int]] = (0, 4), p: float = 1.0):
		"""
			:param nbs_bits: Number of bits to remove in image. Must be in range [0, 8]. (default: (0, 4))
			:param p: The probability to apply the transform. (default: 1.0)
		"""
		super().__init__(p=p)
		self.nbs_bits = nbs_bits if isinstance(nbs_bits, tuple) else (nbs_bits, nbs_bits)
		self.nbs_bits = tuple(map(int, self.nbs_bits))

	def apply(self, data: Image.Image) -> Image.Image:
		nb_bits = np.random.randint(*self.nbs_bits) if self.nbs_bits[0] != self.nbs_bits[1] else self.nbs_bits[0]
		nb_bits = 8 - nb_bits
		return ImageOps.posterize(data, nb_bits)


class Rescale(ImageTransform[Image.Image, Image.Image]):
	def __init__(self, scales: Union[float, Tuple[float, float]] = 1.0, method: int = Image.NEAREST, p: float = 1.0):
		"""
			Available methods :
				Image.ANTIALIAS, Image.BICUBIC, Image.BILINEAR, Image.BOX, Image.HAMMING, Image.NEAREST
		"""
		super().__init__(p=p)
		self.scales = scales
		self.method = method

	def apply(self, data: Image.Image) -> Image.Image:
		scale = np.random.uniform(*self.scales)
		scale -= 1
		scale *= (1.0 if scale <= 0.0 else 0.25)
		size = data.size
		crop = (scale * size[0], scale * size[1], size[0] * (1 - scale), size[1] * (1 - scale))
		return data.crop(crop).resize(data.size, self.method)


class Rotate(ImageTransform[Image.Image, Image.Image]):
	def __init__(self, angles: Union[float, Tuple[float, float]] = 0.0, p: float = 1.0):
		"""
			Rotate an image using PIL methods.

			:param angles: The float or range values for the angle of rotation.
				Angles must be in degrees and in range [-180, 180].
			:param p: The probability to apply the transform.
		"""
		super().__init__(p=p)
		self.angles = angles if isinstance(angles, tuple) else (angles, angles)

	def apply(self, data: Image.Image) -> Image.Image:
		angle = np.random.uniform(*self.angles)
		return data.rotate(angle)


class Sharpness(ImageTransform[Image.Image, Image.Image]):
	def __init__(self, levels: Union[float, Tuple[float, float]] = 0.5, p: float = 1.0):
		"""
			Sharp an image.
			Levels must be in range [-1, 1]. -1 means a blurred image, 1 means a sharpened image and 0 does not change the image.
		"""
		super().__init__(p=p)
		self.enhance = _Enhance(method=ImageEnhance.Sharpness, levels=levels, p=1.0)

	def apply(self, data: Image.Image) -> Image.Image:
		return self.enhance.apply(data)


class ShearX(ImageTransform[Image.Image, Image.Image]):
	def __init__(self, shears: Union[float, Tuple[float, float]] = 0.0, p: float = 1.0):
		"""
			:param shears: The float or range values for the shear parameter.
			:param p: The probability to apply the transform.
		"""
		super().__init__(p=p)
		self.shears = shears if isinstance(shears, tuple) else (shears, shears)

	def apply(self, data: Image.Image) -> Image.Image:
		shear = np.random.uniform(*self.shears)
		return data.transform(data.size, Image.AFFINE, (1, shear, 0, 0, 1, 0))


class ShearY(ImageTransform[Image.Image, Image.Image]):
	def __init__(self, shears: Union[float, Tuple[float, float]] = 0.0, p: float = 1.0):
		"""
			:param shears: The float or range values for the shear parameter.
			:param p: The probability to apply the transform.
		"""
		super().__init__(p=p)
		self.shears = shears if isinstance(shears, tuple) else (shears, shears)

	def apply(self, data: Image.Image) -> Image.Image:
		shear = np.random.uniform(*self.shears)
		return data.transform(data.size, Image.AFFINE, (1, 0, 0, shear, 1, 0))


class Smooth(ImageTransform[Image.Image, Image.Image]):
	def apply(self, data: Image.Image) -> Image.Image:
		return data.filter(ImageFilter.SMOOTH)


class Solarize(ImageTransform[Image.Image, Image.Image]):
	def __init__(self, thresholds: Union[int, Tuple[int, int]] = (0, 256), p: float = 1.0):
		"""
			Invert pixel values above a specific threshold.

			:param thresholds: The int or range values for the threshold parameter.
				All pixel below this value are inverted.
				Must be in range [0, 256].
			:param p: The probability to apply the transform.
		"""
		super().__init__(p=p)
		self.thresholds = thresholds if isinstance(thresholds, tuple) else (thresholds, thresholds)
		self.thresholds = tuple(map(int, self.thresholds))

	def apply(self, data: Image.Image) -> Image.Image:
		threshold = np.random.randint(*self.thresholds) if self.thresholds[0] != self.thresholds[1] else self.thresholds[0]
		threshold = 256 - threshold
		return ImageOps.solarize(data, threshold)


class TranslateX(ImageTransform[Image.Image, Image.Image]):
	def __init__(self, deltas: Union[float, Tuple[float, float]] = 0.0, p: float = 1.0):
		"""
			:param deltas: Ratios min and max for translate image along X axis.
				If equal to -1 or 1, the image is entirely shifted becomes black.
			:param p: Probability to apply the augmentation.
		"""
		super().__init__(p=p)
		self.deltas = deltas if isinstance(deltas, tuple) else (deltas, deltas)

	def apply(self, data: Image.Image) -> Image.Image:
		delta = np.random.uniform(*self.deltas)
		delta = np.floor(delta * data.size[0])
		return data.transform(data.size, Image.AFFINE, (1, 0, delta, 0, 1, 0))


class TranslateY(ImageTransform[Image.Image, Image.Image]):
	def __init__(self, deltas: Union[float, Tuple[float, float]] = 0.0, p: float = 1.0):
		"""
			:param deltas: Ratios min and max for translate image along Y axis.
				If equal to -1 or 1, the image is entirely shifted becomes black.
			:param p: Probability to apply the augmentation.
		"""
		super().__init__(p=p)
		self.deltas = deltas if isinstance(deltas, tuple) else (deltas, deltas)

	def apply(self, data: Image.Image) -> Image.Image:
		delta = np.random.uniform(*self.deltas)
		delta = np.floor(delta * data.size[1])
		return data.transform(data.size, Image.AFFINE, (1, 0, 0, 0, 1, delta))


class VerticalFlip(ImageTransform[Image.Image, Image.Image]):
	def __init__(self, p: float = 1.0):
		super().__init__(p=p)
		self.flip_v = RandomVerticalFlip(1.0)

	def apply(self, data: Image.Image) -> Image.Image:
		return self.flip_v(data)
