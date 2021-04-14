
# import numpy as np
import torch

from unittest import TestCase, main
from mlu.transforms.waveform.time_stretch_pad_crop import TimeStretchPadCrop


class TestStretch(TestCase):
	def test_stretch_pad_crop(self):
		# ADS input example
		# note : transforms are not supposed to work with numpy arrays !
		# x1 = np.random.rand(320000)
		x2 = torch.rand(320000)

		stretch = TimeStretchPadCrop()

		# xa = stretch(x1)
		# self.assertEqual(x1.shape, xa.shape)
		xa = stretch(x2)
		self.assertEqual(x2.shape, xa.shape)

	def test_2(self):
		length = torch.randint(low=1, high=100, size=()).item()
		w = torch.ones(10, length)
		start_shape = w.shape

		stretch = TimeStretchPadCrop()

		for _ in range(10):
			w = stretch(w)

		self.assertEqual(w.shape, start_shape)


if __name__ == '__main__':
	main()
