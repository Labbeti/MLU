
import time
import torch
import tqdm
import unittest

from torch import Tensor
from torch.nn.functional import pad
from unittest import TestCase
from mlu.transforms.base import WaveformTransform


class PadV1(WaveformTransform):
	def __init__(
		self,
		target_length: int,
		align: str = 'left',
		fill_value: float = 0.0,
		dim: int = -1,
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
		super().__init__(p=p)
		self.target_length = target_length
		self.align = align
		self.fill_value = fill_value
		self.dim = dim

	def process(self, waveform: Tensor) -> Tensor:
		if self.align == 'left':
			return self.pad_align_left(waveform)
		elif self.align == 'right':
			return self.pad_align_right(waveform)
		elif self.align == 'center':
			return self.pad_align_center(waveform)
		elif self.align == 'random':
			return self.pad_align_random(waveform)
		else:
			raise ValueError(f'Unknown alignment "{self.align}". Must be one of {str(["left", "right", "center", "random"])}.')

	def pad_align_left(self, data: Tensor) -> Tensor:
		"""
			Pad with left-alignment by adding zeros to right.

			:param data: The original waveform.
		"""
		if self.target_length > data.shape[self.dim]:
			missing = self.target_length - data.shape[self.dim]

			shape_zeros = list(data.shape)
			shape_zeros[self.dim] = missing

			data = torch.cat((
				data,
				torch.full(shape_zeros, self.fill_value, dtype=data.dtype, device=data.device),
			), dim=self.dim)
			data = data.contiguous()
		return data

	def pad_align_right(self, data: Tensor) -> Tensor:
		"""
			Pad with right-alignment by adding zeros to left.

			:param data: The original waveform.
		"""
		if self.target_length > data.shape[self.dim]:
			missing = self.target_length - data.shape[self.dim]

			shape_zeros = list(data.shape)
			shape_zeros[self.dim] = missing

			data = torch.cat((
				torch.full(shape_zeros, self.fill_value, dtype=data.dtype, device=data.device),
				data
			), dim=self.dim)
			data = data.contiguous()
		return data

	def pad_align_center(self, data: Tensor) -> Tensor:
		"""
			Pad with center-alignment by adding half of zeros to left and the other half to right.

			:param data: The original waveform.
		"""
		if self.target_length > data.shape[self.dim]:
			missing = self.target_length - data.shape[self.dim]

			missing_left = missing // 2 + missing % 2
			missing_right = missing // 2

			shape_zeros_left = list(data.shape)
			shape_zeros_left[self.dim] = missing_left

			shape_zeros_right = list(data.shape)
			shape_zeros_right[self.dim] = missing_right

			data = torch.cat((
				torch.full(shape_zeros_left, self.fill_value, dtype=data.dtype, device=data.device),
				data,
				torch.full(shape_zeros_right, self.fill_value, dtype=data.dtype, device=data.device)
			), dim=self.dim)
			data = data.contiguous()
		return data

	def pad_align_random(self, data: Tensor) -> Tensor:
		"""
			Pad with right-alignment by adding zeros randomly to left and right.

			:param data: The original waveform.
		"""
		if self.target_length > data.shape[self.dim]:
			missing = self.target_length - data.shape[self.dim]

			missing_left = torch.randint(low=0, high=missing, size=()).item()
			missing_right = missing - missing_left

			shape_zeros_left = list(data.shape)
			shape_zeros_left[self.dim] = missing_left

			shape_zeros_right = list(data.shape)
			shape_zeros_right[self.dim] = missing_right

			data = torch.cat((
				torch.full(shape_zeros_left, self.fill_value, dtype=data.dtype, device=data.device),
				data,
				torch.full(shape_zeros_right, self.fill_value, dtype=data.dtype, device=data.device),
			), dim=self.dim)
			data = data.contiguous()
		return data


class PadV2(WaveformTransform):
	def __init__(
		self,
		target_length: int,
		align: str = 'left',
		fill_value: float = 0.0,
		dim: int = -1,
		mode: str = 'constant',
		p: float = 1.0,
	):
		super().__init__(p=p)
		self.target_length = target_length
		self.align = align
		self.fill_value = fill_value
		self.dim = dim
		self.mode = mode

	def process(self, waveform: Tensor) -> Tensor:
		if self.align == 'left':
			return self.pad_align_left(waveform)
		elif self.align == 'right':
			return self.pad_align_right(waveform)
		elif self.align == 'center':
			return self.pad_align_center(waveform)
		elif self.align == 'random':
			return self.pad_align_random(waveform)
		else:
			raise ValueError(f'Unknown alignment "{self.align}". Must be one of {str(["left", "right", "center", "random"])}.')

	def pad_align_left(self, x: Tensor) -> Tensor:
		# Note: pad_seq : [pad_left_dim_-1, pad_right_dim_-1, pad_left_dim_-2, pad_right_dim_-2, ...)
		idx = len(x.shape) - (self.dim % len(x.shape)) - 1
		pad_seq = [0 for _ in range(len(x.shape) * 2)]

		missing = max(self.target_length - x.shape[self.dim], 0)
		pad_seq[idx * 2 + 1] = missing

		x = pad(x, pad_seq, mode=self.mode, value=self.fill_value)
		return x

	def pad_align_right(self, x: Tensor) -> Tensor:
		idx = len(x.shape) - (self.dim % len(x.shape)) - 1
		pad_seq = [0 for _ in range(len(x.shape) * 2)]

		missing = max(self.target_length - x.shape[self.dim], 0)
		pad_seq[idx * 2] = missing

		x = pad(x, pad_seq, mode=self.mode, value=self.fill_value)
		return x

	def pad_align_center(self, x: Tensor) -> Tensor:
		idx = len(x.shape) - (self.dim % len(x.shape)) - 1
		pad_seq = [0 for _ in range(len(x.shape) * 2)]

		missing = max(self.target_length - x.shape[self.dim], 0)
		missing_left = missing // 2 + missing % 2
		missing_right = missing // 2

		pad_seq[idx * 2] = missing_left
		pad_seq[idx * 2 + 1] = missing_right

		x = pad(x, pad_seq, mode=self.mode, value=self.fill_value)
		return x

	def pad_align_random(self, x: Tensor) -> Tensor:
		idx = len(x.shape) - (self.dim % len(x.shape)) - 1
		pad_seq = [0 for _ in range(len(x.shape) * 2)]

		missing = max(self.target_length - x.shape[self.dim], 0)
		missing_left = torch.randint(low=0, high=missing + 1, size=()).item()
		missing_right = missing - missing_left

		pad_seq[idx * 2] = missing_left
		pad_seq[idx * 2 + 1] = missing_right

		x = pad(x, pad_seq, mode=self.mode, value=self.fill_value)
		return x


class TestPadVersions(TestCase):
	def test_compare_pad(self):
		target_length = torch.randint(low=5, high=20, size=()).item()
		pad_v1 = PadV1(target_length, 'left', 0)
		pad_v2 = PadV2(target_length, 'left', 0)

		x = torch.rand(5)

		out_v1 = pad_v1(x)
		out_v2 = pad_v2(x)
		expected = torch.cat((x, torch.zeros(target_length - 5)))

		self.assertEqual(out_v1.shape, out_v2.shape)
		self.assertTrue(out_v1.eq(out_v2).all(), f'Diff: {out_v1} and {out_v2}')

		self.assertEqual(out_v1.shape, expected.shape)
		self.assertTrue(out_v1.eq(expected).all())

	def test_compare_pad_2(self):
		x = torch.rand(5, 2, 3)
		dim = torch.randint(low=-len(x.shape), high=len(x.shape), size=()).item()
		target_length = torch.randint(low=x.shape[dim], high=20, size=()).item()

		for align in ['left', 'right', 'center']:
			pad_v1 = PadV1(target_length, align, 0, dim)
			pad_v2 = PadV2(target_length, align, 0, dim)

			out_v1 = pad_v1(x)
			out_v2 = pad_v2(x)

			self.assertEqual(out_v1.shape, out_v2.shape)
			self.assertTrue(out_v1.eq(out_v2).all(), f'Diff: {out_v1} and {out_v2}')

	def test_compare_random(self):
		align = 'random'
		dim = 1
		x = torch.rand(5, 3)

		max_length = x.shape[dim] * 3
		target_length = torch.randint(low=x.shape[dim], high=max_length, size=()).item()

		pad_v1 = PadV1(target_length, align, 0, dim)
		pad_v2 = PadV2(target_length, align, 0, dim)

		out_v1 = pad_v1(x)
		out_v2 = pad_v2(x)

		self.assertEqual(out_v1.shape, out_v2.shape)
		self.assertEqual((out_v1 == 0.0).sum(), (out_v2 == 0.0).sum())

	def test_limits(self):
		target_length = 10
		dim = 0
		x = torch.ones(0)
		expected = torch.zeros(target_length)

		for align in ['left', 'right', 'center', 'random']:
			pad_v1 = PadV1(target_length, align, 0, dim)
			pad_v2 = PadV2(target_length, align, 0, dim)

			out_v1 = pad_v1(x)
			out_v2 = pad_v2(x)

			self.assertTrue(out_v1.eq(expected).all())
			self.assertTrue(out_v2.eq(expected).all())

	def test_large(self):
		durations_v1 = []
		durations_v2 = []
		n_tests = 10000

		for _ in tqdm.trange(n_tests):
			size = torch.randint(low=1, high=2, size=(1,)).tolist()
			shape = torch.randint(low=1, high=100000, size=size)
			x = torch.rand(*shape)

			target_length = torch.randint(low=shape.min(), high=1000000, size=()).item()
			align = ['left', 'right', 'center', 'random'][torch.randint(low=0, high=4, size=()).item()]
			dim = torch.randint(low=-len(x.shape), high=len(x.shape), size=()).item()
			fill_value = torch.rand(1).item()

			start = time.time()
			pad_v1 = PadV1(target_length, align, fill_value, dim)
			out_v1 = pad_v1(x)
			durations_v1.append(time.time() - start)

			start = time.time()
			pad_v2 = PadV2(target_length, align, fill_value, dim)
			out_v2 = pad_v2(x)
			durations_v2.append(time.time() - start)

			self.assertEqual(out_v1.shape, out_v2.shape)
			self.assertEqual((out_v1 == 0.0).sum(), (out_v2 == 0.0).sum())
			if align != 'random':
				self.assertTrue(out_v1.eq(out_v2).all())

		durations_v1 = torch.as_tensor(durations_v1)
		durations_v2 = torch.as_tensor(durations_v2)
		print()
		print(f'Mean dur V1: {durations_v1.mean():.5e} +/- {durations_v1.std():.5e}')
		print(f'Mean dur V2: {durations_v2.mean():.5e} +/- {durations_v2.std():.5e}')


if __name__ == '__main__':
	unittest.main()
