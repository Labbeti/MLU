
import numpy as np
import torch
import torchvision

from mlu.transforms.converters import ToList, ToPIL, ToNumpy, ToTensor
from PIL import Image
from unittest import TestCase, main


class TestConvert(TestCase):
	def test_tensor_conversions(self):
		# Black image (channel, height, width)
		data = torch.zeros(3, 64, 32)

		to_ten = ToTensor(device=data.device)
		to_num = ToNumpy()
		to_lis = ToList()
		to_pil = ToPIL()

		to_base = to_ten

		for to in [to_ten, to_num, to_lis, to_pil]:
			other = to_base(to(data))

			self.assertEqual(data.shape, other.shape,
				f'Mismatch shapes for conversion {to_base.__class__.__name__}({to.__class__.__name__}(x)) == x')
			self.assertTrue(data.eq(other).all(),
				f'Mismatch values for conversion {to_base.__class__.__name__}({to.__class__.__name__}(x)) == x')

	def test_numpy_conversions(self):
		# Black image (width, height, channel)
		data = np.zeros((32, 64, 3))

		to_ten = ToTensor()
		to_num = ToNumpy()
		to_lis = ToList()
		to_pil = ToPIL()

		to_base = to_num

		for to in [to_ten, to_num, to_lis, to_pil]:
			other = to_base(to(data))

			self.assertEqual(data.shape, other.shape,
				f'Mismatch shapes for conversion {to_base.__class__.__name__}({to.__class__.__name__}(x)) == x')
			self.assertTrue((data == other).all(),
				f'Mismatch values for conversion {to_base.__class__.__name__}({to.__class__.__name__}(x)) == x')

	def test_list_conversions(self):
		# Black image (width, height, channel)
		data = [[[0 for _ in range(3)] for _ in range(64)] for _ in range(32)]

		to_ten = ToTensor()
		to_num = ToNumpy()
		to_lis = ToList()
		to_pil = ToPIL()

		to_base = to_lis

		for to in [to_ten, to_num, to_lis, to_pil]:
			other = to_base(to(data))
			data_shape = _get_list_shape(data)
			other_shape = _get_list_shape(other)

			self.assertEqual(data_shape, other_shape,
				f'Mismatch shapes for conversion {to_base.__class__.__name__}({to.__class__.__name__}(x)) == x')
			self.assertTrue(data == other,
				f'Mismatch values for conversion {to_base.__class__.__name__}({to.__class__.__name__}(x)) == x')

	def test_pil_conversions(self):
		to_ten = ToTensor()
		to_num = ToNumpy()
		to_lis = ToList()
		to_pil = ToPIL()

		# Black image (width, height)
		data = Image.new('RGB', (32, 64), color='black')
		to_base = to_pil

		for to in [to_ten, to_num, to_lis, to_pil]:
			other = to_base(to(data))

			self.assertEqual(data.size, other.size,
				f'Mismatch shapes for conversion {to_base.__class__.__name__}({to.__class__.__name__}(x)) == x')
			self.assertTrue(data == other,
				f'Mismatch values for conversion {to_base.__class__.__name__}({to.__class__.__name__}(x)) == x')


class TestCompat(TestCase):
	def test_to_tensor(self):
		to_tens_mlu = ToTensor()
		to_tens_tvi = torchvision.transforms.ToTensor()

		# Black image (width, height, 3)
		data = Image.new('RGB', (32, 64), color='black')

		other_mlu = to_tens_mlu(data)
		other_tvi = to_tens_tvi(data)

		self.assertEqual(other_mlu.shape, other_tvi.shape,
			f'Mismatch shapes for conversion {to_tens_mlu.__class__.__name__}(x) == {to_tens_tvi.__class__.__name__}(x)')
		self.assertTrue(other_mlu.eq(other_tvi).all(),
			f'Mismatch values for conversion {to_tens_mlu.__class__.__name__}(x) == {to_tens_tvi.__class__.__name__}(x)')

	def test_to_pil(self):
		to_pil_mlu = ToPIL()
		to_pil_tvi = torchvision.transforms.ToPILImage()

		# Black image (channel, height, width)
		data = torch.zeros(3, 64, 32)

		other_mlu = to_pil_mlu(data)
		other_tvi = to_pil_tvi(data)

		self.assertEqual(other_mlu.size, other_tvi.size,
			f'Mismatch shapes for conversion {to_pil_mlu.__class__.__name__}(x) == {to_pil_tvi.__class__.__name__}(x)')
		self.assertTrue(other_mlu == other_tvi,
			f'Mismatch values for conversion {to_pil_mlu.__class__.__name__}(x) == {to_pil_tvi.__class__.__name__}(x)')


def _get_list_shape(lst: list) -> tuple:
	elt = lst
	shape = []
	while isinstance(elt, list):
		shape.append(len(elt))
		if len(elt) > 0:
			elt = elt[0]
		else:
			break
	return tuple(shape)


if __name__ == '__main__':
	main()
