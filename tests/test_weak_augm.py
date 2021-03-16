
import torch
import unittest

from unittest import TestCase

from mlu.transforms import StretchPadCrop


class TestAugm(TestCase):
	def test_stretch_pad_crop(self):
		x = torch.rand(320000)  # ADS input example
		stretch = StretchPadCrop()
		xa = stretch(x)
		self.assertEqual(x.shape, xa.shape)


if __name__ == "__main__":
	unittest.main()
