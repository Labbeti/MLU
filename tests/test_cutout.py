
import torch

from mlu.transforms.convert import ToPIL
from mlu.transforms.image.pil import CutOutImgPIL
from mlu.transforms.image.tensor import CutOutImg as CutOutImgTen
from unittest import TestCase, main


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


if __name__ == "__main__":
	main()
