
import numpy as np
import torch

from mlu.transforms.conversion.conversion import ToList, ToPIL, ToNumpy, ToTensor
from unittest import TestCase, main


class TestConverts(TestCase):
	def test_tensor_conversions(self):
		data = torch.zeros(32, 64, 3).cuda()

		to_ten = ToTensor(device=data.device)
		to_num = ToNumpy()
		to_lis = ToList()
		to_pil = ToPIL()

		to_base = to_ten

		for to in [to_ten, to_num, to_lis, to_pil]:
			other = to_base(to(data))
			self.assertTrue(data.eq(other).all(), "Assertion false for conversion \"{}({}(x)) == x\".".format(
				str(to_base.__class__.__name__), str(to.__class__.__name__)))

	def test_numpy_conversions(self):
		to_ten = ToTensor()
		to_num = ToNumpy()
		to_lis = ToList()
		to_pil = ToPIL()

		data = np.zeros((32, 64, 3))
		to_base = to_num

		for to in [to_ten, to_num, to_lis, to_pil]:
			other = to_base(to(data))
			self.assertTrue((data == other).all(), "Assertion false for conversion \"{}({}(x)) == x\".".format(
				str(to_base.__class__.__name__), str(to.__class__.__name__)))

	def test_list_conversions(self):
		to_ten = ToTensor()
		to_num = ToNumpy()
		to_lis = ToList()
		to_pil = ToPIL()

		data = [[[0 for _ in range(3)] for _ in range(64)] for _ in range(32)]
		to_base = to_lis

		for to in [to_ten, to_num, to_lis, to_pil]:
			other = to_base(to(data))
			self.assertTrue(data == other, "Assertion false for conversion \"{}({}(x)) == x\".".format(
				str(to_base.__class__.__name__), str(to.__class__.__name__)))

	def test_pil_conversions(self):
		to_ten = ToTensor()
		to_num = ToNumpy()
		to_lis = ToList()
		to_pil = ToPIL()

		data = to_pil(torch.zeros(32, 64, 3))
		to_base = to_pil

		for to in [to_ten, to_num, to_lis, to_pil]:
			other = to_base(to(data))
			self.assertTrue(data == other, "Assertion false for conversion \"{}({}(x)) == x\".".format(
				str(to_base.__class__.__name__), str(to.__class__.__name__)))


if __name__ == "__main__":
	main()
