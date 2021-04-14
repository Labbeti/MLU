
import torch
import unittest

from mlu.transforms.image import Standardize
from torchvision.transforms import Normalize
from unittest import TestCase


class TestStd(TestCase):
	def test_compat(self):
		data = torch.rand(3, 64, 32)

		means = (0.4, 0.6, 0.5)
		stds = (0.1, 0.05, 0.1)

		std_mlu = Standardize(means, stds, channel_dim=0)
		std_tvi = Normalize(means, stds)

		data_mlu = std_mlu(data)
		data_tvi = std_tvi(data)

		self.assertTrue(data_mlu.eq(data_tvi).all(),
			f'Mismatch values of Standardized images.')


if __name__ == '__main__':
	unittest.main()
