
import torch

from mlu.transforms.waveform.stretch_pad_crop import StretchPadCrop
from unittest import TestCase, main


class TestStretch(TestCase):
	def test_stretch_pad_crop(self):
		x = torch.rand(320000)  # ADS input example
		stretch = StretchPadCrop()
		xa = stretch(x)
		self.assertEqual(x.shape, xa.shape)


if __name__ == "__main__":
	main()
