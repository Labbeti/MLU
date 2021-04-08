
import time
import torch
import tqdm
import unittest

from torch import Tensor
from torch.nn.functional import pad
from unittest import TestCase
from mlu.transforms.base import WaveformTransform
from mlu.transforms.waveform.pad import Pad


class PadV2(WaveformTransform):
	def __init__(self, target_length: int, align: str = "left", fill_value: float = 0.0, dim: int = -1, p: float = 1.0):
		super().__init__(p=p)
		self.target_length = target_length
		self.align = align
		self.fill_value = fill_value
		self.dim = dim

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

	def pad_align_left(self, x: Tensor) -> Tensor:
		# Note: pad_seq : [pad_left_dim_-1, pad_right_dim_-1, pad_left_dim_-2, pad_right_dim_-2, ...)
		idx = len(x.shape) - (self.dim % len(x.shape)) - 1
		pad_seq = [0 for _ in range(len(x.shape) * 2)]

		missing = max(self.target_length - x.shape[self.dim], 0)
		pad_seq[idx * 2 + 1] = missing

		x = pad(x, pad_seq, mode="constant", value=self.fill_value)
		return x

	def pad_align_right(self, x: Tensor) -> Tensor:
		idx = len(x.shape) - (self.dim % len(x.shape)) - 1
		pad_seq = [0 for _ in range(len(x.shape) * 2)]

		missing = max(self.target_length - x.shape[self.dim], 0)
		pad_seq[idx * 2] = missing

		x = pad(x, pad_seq, mode="constant", value=self.fill_value)
		return x

	def pad_align_center(self, x: Tensor) -> Tensor:
		idx = len(x.shape) - (self.dim % len(x.shape)) - 1
		pad_seq = [0 for _ in range(len(x.shape) * 2)]

		missing = max(self.target_length - x.shape[self.dim], 0)
		missing_left = missing // 2 + missing % 2
		missing_right = missing // 2

		pad_seq[idx * 2] = missing_left
		pad_seq[idx * 2 + 1] = missing_right

		x = pad(x, pad_seq, mode="constant", value=self.fill_value)
		return x

	def pad_align_random(self, x: Tensor) -> Tensor:
		idx = len(x.shape) - (self.dim % len(x.shape)) - 1
		pad_seq = [0 for _ in range(len(x.shape) * 2)]

		missing = max(self.target_length - x.shape[self.dim], 0)
		missing_left = torch.randint(low=0, high=missing + 1, size=()).item()
		missing_right = missing - missing_left

		pad_seq[idx * 2] = missing_left
		pad_seq[idx * 2 + 1] = missing_right

		x = pad(x, pad_seq, mode="constant", value=self.fill_value)
		return x


class TestPadVersions(TestCase):
	def test_compare_pad(self):
		target_length = torch.randint(low=5, high=20, size=()).item()
		pad_v1 = Pad(target_length, "left", 0)
		pad_v2 = PadV2(target_length, "left", 0)

		x = torch.rand(5)

		out_v1 = pad_v1(x)
		out_v2 = pad_v2(x)
		expected = torch.cat((x, torch.zeros(target_length - 5)))

		self.assertEqual(out_v1.shape, out_v2.shape)
		self.assertTrue(out_v1.eq(out_v2).all(), f"Diff: {out_v1} and {out_v2}")

		self.assertEqual(out_v1.shape, expected.shape)
		self.assertTrue(out_v1.eq(expected).all())

	def test_compare_pad_2(self):
		x = torch.rand(5, 2, 3)
		dim = torch.randint(low=-len(x.shape), high=len(x.shape), size=()).item()
		target_length = torch.randint(low=x.shape[dim], high=20, size=()).item()

		for align in ["left", "right", "center"]:
			pad_v1 = Pad(target_length, align, 0, dim)
			pad_v2 = PadV2(target_length, align, 0, dim)

			out_v1 = pad_v1(x)
			out_v2 = pad_v2(x)

			self.assertEqual(out_v1.shape, out_v2.shape)
			self.assertTrue(out_v1.eq(out_v2).all(), f"Diff: {out_v1} and {out_v2}")

	def test_compare_random(self):
		align = "random"
		dim = 1
		x = torch.rand(5, 3)

		max_length = x.shape[dim] * 3
		target_length = torch.randint(low=x.shape[dim], high=max_length, size=()).item()

		pad_v1 = Pad(target_length, align, 0, dim)
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

		for align in ["left", "right", "center", "random"]:
			pad_v1 = Pad(target_length, align, 0, dim)
			pad_v2 = PadV2(target_length, align, 0, dim)

			out_v1 = pad_v1(x)
			out_v2 = pad_v2(x)

			self.assertTrue(out_v1.eq(expected).all())
			self.assertTrue(out_v2.eq(expected).all())

	def test_large(self):
		durations_v1 = []
		durations_v2 = []
		num_tests = 1000

		for _ in tqdm.trange(num_tests):
			size = torch.randint(low=1, high=2, size=(1,)).tolist()
			shape = torch.randint(low=1, high=400000, size=size)
			x = torch.rand(*shape)

			target_length = torch.randint(low=shape.min(), high=1000000, size=()).item()
			align = ["left", "right", "center", "random"][torch.randint(low=0, high=4, size=()).item()]
			dim = torch.randint(low=-len(x.shape), high=len(x.shape), size=()).item()
			fill_value = torch.rand(1).item()

			start = time.time()
			pad_v1 = Pad(target_length, align, fill_value, dim)
			out_v1 = pad_v1(x)
			durations_v1.append(time.time() - start)

			start = time.time()
			pad_v2 = PadV2(target_length, align, fill_value, dim)
			out_v2 = pad_v2(x)
			durations_v2.append(time.time() - start)

			self.assertEqual(out_v1.shape, out_v2.shape)
			self.assertEqual((out_v1 == 0.0).sum(), (out_v2 == 0.0).sum())
			if align != "random":
				self.assertTrue(out_v1.eq(out_v2).all())

		durations_v1 = torch.as_tensor(durations_v1)
		durations_v2 = torch.as_tensor(durations_v2)
		print()
		print(f"Mean dur V1: {durations_v1.mean():.5e} +/- {durations_v1.std():.5e}")
		print(f"Mean dur V2: {durations_v2.mean():.5e} +/- {durations_v2.std():.5e}")


if __name__ == "__main__":
	unittest.main()
