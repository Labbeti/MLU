
import torch

from mlu.transforms.base import WaveformTransform

from torch import Tensor


class PadAlignLeft(WaveformTransform):
	def __init__(
		self,
		target_length: int,
		fill_value: float = 0.0,
		dim: int = -1,
		p: float = 1.0,
	):
		"""
			Pad with left-alignment by adding zeros to right.

			:param target_length: The target length of the waveform.
			:param fill_value: The value used to pad the waveform. (default: 0.0)
			:param dim: The dimension to apply the padding. (default: -1)
			:param p: The probability to apply the transform. (default: 1.0)
		"""
		super().__init__(p)
		self.target_length = target_length
		self.fill_value = fill_value
		self.dim = dim

	def process(self, waveform: Tensor) -> Tensor:
		waveform = pad_align_left(waveform, self.target_length, self.fill_value, self.dim)
		return waveform


class PadAlignRight(WaveformTransform):
	def __init__(
		self,
		target_length: int,
		fill_value: float = 0.0,
		dim: int = -1,
		p: float = 1.0,
	):
		"""
			Pad with right-alignment by adding zeros to left.

			:param target_length: The target length of the waveform.
			:param fill_value: The value used to pad the waveform. (default: 0.0)
			:param dim: The dimension to apply the padding. (default: -1)
			:param p: The probability to apply the transform. (default: 1.0)
		"""
		super().__init__(p)
		self.target_length = target_length
		self.fill_value = fill_value
		self.dim = dim

	def process(self, waveform: Tensor) -> Tensor:
		waveform = pad_align_right(waveform, self.target_length, self.fill_value, self.dim)
		return waveform


class PadAlignCenter(WaveformTransform):
	def __init__(
		self,
		target_length: int,
		fill_value: float = 0.0,
		dim: int = -1,
		p: float = 1.0,
	):
		"""
			Pad with center-alignment by adding half of zeros to left and the other half to right.

			:param target_length: The target length of the waveform.
			:param fill_value: The value used to pad the waveform. (default: 0.0)
			:param dim: The dimension to apply the padding. (default: -1)
			:param p: The probability to apply the transform. (default: 1.0)
		"""
		super().__init__(p)
		self.target_length = target_length
		self.fill_value = fill_value
		self.dim = dim

	def process(self, waveform: Tensor) -> Tensor:
		waveform = pad_align_center(waveform, self.target_length, self.fill_value, self.dim)
		return waveform


class PadAlignRandom(WaveformTransform):
	def __init__(
		self,
		target_length: int,
		fill_value: float = 0.0,
		dim: int = -1,
		p: float = 1.0,
	):
		"""
			Pad with right-alignment by adding zeros randomly to left and right.

			:param target_length: The target length of the waveform.
			:param fill_value: The value used to pad the waveform. (default: 0.0)
			:param dim: The dimension to apply the padding. (default: -1)
			:param p: The probability to apply the transform. (default: 1.0)
		"""
		super().__init__(p)
		self.target_length = target_length
		self.fill_value = fill_value
		self.dim = dim

	def process(self, waveform: Tensor) -> Tensor:
		waveform = pad_align_random(waveform, self.target_length, self.fill_value, self.dim)
		return waveform


class Pad(WaveformTransform):
	def __init__(
		self,
		target_length: int,
		fill_value: float = 0.0,
		dim: int = -1,
		align: str = "left",
		p: float = 1.0,
	):
		"""
			Pad with alignment by adding zeros to left or right.

			:param target_length: The target length of the waveform.
			:param fill_value: The value used to pad the waveform. (default: 0.0)
			:param dim: The dimension to apply the padding. (default: -1)
			:param align: The waveform alignment. Determine if zeros will be added to left or right.
				(available alignment are : 'left', 'right', 'center' and 'random').
				(default: 'left')
			:param p: The probability to apply the transform. (default: 1.0)
		"""
		super().__init__(p)
		self.target_length = target_length
		self.fill_value = fill_value
		self.dim = dim
		self.align = align

	def process(self, waveform: Tensor) -> Tensor:
		waveform = pad(waveform, self.target_length, self.fill_value, self.dim, self.align)
		return waveform


def pad_align_left(waveform: Tensor, target_length: int, fill_value: float, dim: int) -> Tensor:
	"""
		Pad with left-alignment by adding zeros to right.

		:param waveform: The original waveform.
		:param target_length: The target length of the waveform.
		:param fill_value: The value used to pad the waveform.
		:param dim: The dimension to apply the padding.
	"""
	if target_length > waveform.shape[dim]:
		missing = target_length - waveform.shape[dim]

		shape_zeros = list(waveform.shape)
		shape_zeros[dim] = missing

		waveform = torch.cat((
			waveform,
			torch.full(shape_zeros, fill_value, dtype=waveform.dtype, device=waveform.device),
		), dim=dim)
	return waveform


def pad_align_right(waveform: Tensor, target_length: int, fill_value: float, dim: int) -> Tensor:
	"""
		Pad with right-alignment by adding zeros to left.

		:param waveform: The original waveform.
		:param target_length: The target length of the waveform.
		:param fill_value: The value used to pad the waveform.
		:param dim: The dimension to apply the padding.
	"""
	if target_length > waveform.shape[dim]:
		missing = target_length - waveform.shape[dim]

		shape_zeros = list(waveform.shape)
		shape_zeros[dim] = missing

		waveform = torch.cat((
			torch.full(shape_zeros, fill_value, dtype=waveform.dtype, device=waveform.device),
			waveform
		), dim=dim)
	return waveform


def pad_align_center(waveform: Tensor, target_length: int, fill_value: float, dim: int) -> Tensor:
	"""
		Pad with center-alignment by adding half of zeros to left and the other half to right.

		:param waveform: The original waveform.
		:param target_length: The target length of the waveform.
		:param fill_value: The value used to pad the waveform.
		:param dim: The dimension to apply the padding.
	"""
	if target_length > waveform.shape[dim]:
		missing = target_length - waveform.shape[dim]

		missing_left = missing // 2 + missing % 2
		missing_right = missing // 2

		shape_zeros_left = list(waveform.shape)
		shape_zeros_left[dim] = missing_left

		shape_zeros_right = list(waveform.shape)
		shape_zeros_right[dim] = missing_right

		waveform = torch.cat((
			torch.full(shape_zeros_left, fill_value, dtype=waveform.dtype, device=waveform.device),
			waveform,
			torch.full(shape_zeros_right, fill_value, dtype=waveform.dtype, device=waveform.device)
		), dim=dim)
	return waveform


def pad_align_random(waveform: Tensor, target_length: int, fill_value: float, dim: int) -> Tensor:
	"""
		Pad with right-alignment by adding zeros randomly to left and right.

		:param waveform: The original waveform.
		:param target_length: The target length of the waveform.
		:param fill_value: The value used to pad the waveform.
		:param dim: The dimension to apply the padding.
	"""
	if target_length > waveform.shape[dim]:
		missing = target_length - waveform.shape[dim]

		missing_left = torch.randint(low=0, high=missing, size=()).item()
		missing_right = missing - missing_left

		shape_zeros_left = list(waveform.shape)
		shape_zeros_left[dim] = missing_left

		shape_zeros_right = list(waveform.shape)
		shape_zeros_right[dim] = missing_right

		waveform = torch.cat((
			torch.full(shape_zeros_left, fill_value, dtype=waveform.dtype, device=waveform.device),
			waveform,
			torch.full(shape_zeros_right, fill_value, dtype=waveform.dtype, device=waveform.device),
		), dim=dim)
	return waveform


def pad(waveform: Tensor, target_length: int, fill_value: float, dim: int, align: str) -> Tensor:
	"""
		Pad with alignment by adding zeros to left or right.

		:param waveform: The original waveform.
		:param target_length: The target length of the waveform.
		:param fill_value: The value used to pad the waveform.
		:param dim: The dimension to apply the padding.
		:param align: The waveform alignment. Determine if zeros will be added to left or right.
			(available alignment are : 'left', 'right', 'center' and 'random').
	"""
	if align == "left":
		return pad_align_left(waveform, target_length, fill_value, dim)
	elif align == "right":
		return pad_align_right(waveform, target_length, fill_value, dim)
	elif align == "center":
		return pad_align_center(waveform, target_length, fill_value, dim)
	elif align == "random":
		return pad_align_random(waveform, target_length, fill_value, dim)
	else:
		raise ValueError(f"Unknown alignment '{align}'. Must be one of {str(['left', 'right', 'center', 'random'])}.")
