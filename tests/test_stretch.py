
# import numpy as np
import torch

from unittest import TestCase, main
from mlu.transforms.waveform.stretch_pad_crop import StretchPadCrop


class TestStretch(TestCase):
	def test_stretch_pad_crop(self):
		# ADS input example
		# note : transforms are not supposed to work with numpy arrays !
		# x1 = np.random.rand(320000)
		x2 = torch.rand(320000)

		stretch = StretchPadCrop()

		# xa = stretch(x1)
		# self.assertEqual(x1.shape, xa.shape)
		xa = stretch(x2)
		self.assertEqual(x2.shape, xa.shape)


if __name__ == "__main__":
	main()
