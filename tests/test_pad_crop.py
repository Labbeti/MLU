
import torch

from mlu.transforms.waveform.crop import CropLeft, CropRight, CropCenter
from mlu.transforms.waveform.pad import PadLeft, PadRight, PadCenter, PadRandom
from unittest import TestCase, main


class TestPad(TestCase):
	def test_1(self):
		target = 10
		pad = PadCenter(target)

		x = torch.ones(7)
		out = pad(x)

		self.assertEqual(out.tolist(), [0, 0, 1, 1, 1, 1, 1, 1, 1, 0])

	def test_2(self):
		target = 100
		pad = PadRandom(target)

		x = torch.ones(1, 2, 80)
		out = pad(x)

		self.assertEqual(target, out.shape[-1])

	def test_3(self):
		target = 200
		pad = PadLeft(target)

		x = torch.ones(16, 5, 60)
		out = pad(x)

		self.assertEqual(list(out.shape), [16, 5, target])

	def test_4(self):
		target = 7
		pad = PadRight(target)

		x = torch.ones(2, 1, 4)
		out = pad(x)

		self.assertEqual(out.tolist(), [[[0, 0, 0, 1, 1, 1, 1]], [[0, 0, 0, 1, 1, 1, 1]]])


class TestCut(TestCase):
	def test_1(self):
		target = 100
		cut = CropLeft(target)

		x = torch.zeros(120)
		out = cut(x)

		self.assertEqual(out.shape[-1], target)

	def test_2(self):
		target = 100
		cut = CropRight(target)

		x = torch.zeros(1, 1, 120)
		out = cut(x)

		self.assertEqual(list(out.shape), [1, 1, target])

	def test_3(self):
		target = 100
		cut = CropCenter(target)

		x = torch.zeros(4, 2, 130)
		out = cut(x)

		self.assertEqual(list(out.shape), [4, 2, target])


if __name__ == "__main__":
	main()
