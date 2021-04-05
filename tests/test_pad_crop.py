
import torch
import unittest

from unittest import TestCase
from mlu.transforms.waveform.crop import CropAlignLeft, CropAlignRight, CropAlignCenter
from mlu.transforms.waveform.pad import PadAlignLeft, PadAlignRight, PadAlignCenter, PadAlignRandom


class TestPad(TestCase):
	def test_1(self):
		target = 10
		pad = PadAlignCenter(target)

		x = torch.ones(7)
		out = pad(x)

		self.assertEqual(out.tolist(), [0, 0, 1, 1, 1, 1, 1, 1, 1, 0])

	def test_2(self):
		target = 100
		pad = PadAlignRandom(target)

		x = torch.ones(1, 2, 80)
		out = pad(x)

		self.assertEqual(target, out.shape[-1])

	def test_3(self):
		target = 200
		pad = PadAlignLeft(target)

		x = torch.ones(16, 5, 60)
		out = pad(x)

		self.assertEqual(list(out.shape), [16, 5, target])

	def test_4(self):
		target = 7
		pad = PadAlignRight(target)

		x = torch.ones(2, 1, 4)
		out = pad(x)

		self.assertEqual(out.tolist(), [[[0, 0, 0, 1, 1, 1, 1]], [[0, 0, 0, 1, 1, 1, 1]]])


class TestCut(TestCase):
	def test_1(self):
		target = 100
		cut = CropAlignLeft(target)

		x = torch.zeros(120)
		out = cut(x)

		self.assertEqual(out.shape[-1], target)

	def test_2(self):
		target = 100
		cut = CropAlignRight(target)

		x = torch.zeros(1, 1, 120)
		out = cut(x)

		self.assertEqual(list(out.shape), [1, 1, target])

	def test_3(self):
		target = 100
		cut = CropAlignCenter(target)

		x = torch.zeros(4, 2, 130)
		out = cut(x)

		self.assertEqual(list(out.shape), [4, 2, target])


if __name__ == "__main__":
	unittest.main()
