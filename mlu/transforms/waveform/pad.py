
import torch

from torch import Tensor
from mlu.transforms.base import WaveformTransform


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
		if self.align == "left":
			return self.pad_align_left(waveform)
		elif self.align == "right":
			return self.pad_align_right(waveform)
		elif self.align == "center":
			return self.pad_align_center(waveform)
		elif self.align == "random":
			return self.pad_align_random(waveform)
		else:
			raise ValueError(f"Unknown alignment '{self.align}'. Must be one of {str(['left', 'right', 'center', 'random'])}.")

	def pad_align_left(self, waveform: Tensor) -> Tensor:
		"""
			Pad with left-alignment by adding zeros to right.

			:param waveform: The original waveform.
		"""
		if self.target_length > waveform.shape[self.dim]:
			missing = self.target_length - waveform.shape[self.dim]

			shape_zeros = list(waveform.shape)
			shape_zeros[self.dim] = missing

			waveform = torch.cat((
				waveform,
				torch.full(shape_zeros, self.fill_value, dtype=waveform.dtype, device=waveform.device),
			), dim=self.dim)
		return waveform

	def pad_align_right(self, waveform: Tensor) -> Tensor:
		"""
			Pad with right-alignment by adding zeros to left.

			:param waveform: The original waveform.
		"""
		if self.target_length > waveform.shape[self.dim]:
			missing = self.target_length - waveform.shape[self.dim]

			shape_zeros = list(waveform.shape)
			shape_zeros[self.dim] = missing

			waveform = torch.cat((
				torch.full(shape_zeros, self.fill_value, dtype=waveform.dtype, device=waveform.device),
				waveform
			), dim=self.dim)
		return waveform

	def pad_align_center(self, waveform: Tensor) -> Tensor:
		"""
			Pad with center-alignment by adding half of zeros to left and the other half to right.

			:param waveform: The original waveform.
		"""
		if self.target_length > waveform.shape[self.dim]:
			missing = self.target_length - waveform.shape[self.dim]

			missing_left = missing // 2 + missing % 2
			missing_right = missing // 2

			shape_zeros_left = list(waveform.shape)
			shape_zeros_left[self.dim] = missing_left

			shape_zeros_right = list(waveform.shape)
			shape_zeros_right[self.dim] = missing_right

			waveform = torch.cat((
				torch.full(shape_zeros_left, self.fill_value, dtype=waveform.dtype, device=waveform.device),
				waveform,
				torch.full(shape_zeros_right, self.fill_value, dtype=waveform.dtype, device=waveform.device)
			), dim=self.dim)
		return waveform

	def pad_align_random(self, waveform: Tensor) -> Tensor:
		"""
			Pad with right-alignment by adding zeros randomly to left and right.

			:param waveform: The original waveform.
		"""
		if self.target_length > waveform.shape[self.dim]:
			missing = self.target_length - waveform.shape[self.dim]

			missing_left = torch.randint(low=0, high=missing, size=()).item()
			missing_right = missing - missing_left

			shape_zeros_left = list(waveform.shape)
			shape_zeros_left[self.dim] = missing_left

			shape_zeros_right = list(waveform.shape)
			shape_zeros_right[self.dim] = missing_right

			waveform = torch.cat((
				torch.full(shape_zeros_left, self.fill_value, dtype=waveform.dtype, device=waveform.device),
				waveform,
				torch.full(shape_zeros_right, self.fill_value, dtype=waveform.dtype, device=waveform.device),
			), dim=self.dim)
		return waveform
