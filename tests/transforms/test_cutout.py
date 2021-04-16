
import torch
import unittest

from mlu.transforms.converters import ToPIL
from mlu.transforms.image.pil import CutOutImgPIL
from mlu.transforms.image.tensor import CutOutImg as CutOutImgTen
from mlu.transforms.spectrogram import CutOutSpec
from unittest import TestCase


class TestCutOut(TestCase):
	def test_(self):
		img_ten = torch.rand(32, 64, 3)
		img_pil = ToPIL()(img_ten)

		cut_ten = CutOutImgTen()
		cut_pil = CutOutImgPIL()

		img_ten_res = cut_ten(img_ten)
		img_pil_res = cut_pil(img_pil)

		"""
		plt.figure()
		plt.imshow(img_ten.numpy())
		plt.figure()
		plt.imshow(img_pil)
		plt.figure()
		plt.imshow(img_ten_res.numpy())
		plt.figure()
		plt.imshow(img_pil_res)
		plt.show()
		"""

		self.assertFalse(img_ten.eq(img_ten_res).all())
		self.assertNotEqual(img_pil, img_pil_res)

	def test_cut_out_spec(self):
		transform = CutOutSpec(freq_scales=(1.0, 1.0), time_scales=(0.1, 0.1), fill_value=0.0)

		t = torch.rand(3, 32, 16)
		o = transform(t)

		print('Shapes')
		print(t.shape)
		print(o.shape)

		from matplotlib import pyplot as plt
		plt.figure()
		plt.imshow(t.numpy().T)
		plt.figure()
		plt.imshow(o.numpy().T)
		plt.show(block=False)
		# input("> ")

	def test_cos_2(self):
		transform = CutOutSpec(freq_scales=(0.0, 1.0), time_scales=(0.0, 1.0), fill_value=-1.0)
		t = torch.rand(3, 32, 16)
		o = transform(t)

		self.assertEqual(t.shape, o.shape)
		self.assertGreater(t.sum(), o.sum())
		self.assertGreater(o.eq(-1.0).sum(), 0)


if __name__ == '__main__':
	unittest.main()
