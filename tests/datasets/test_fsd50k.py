
import logging
import os.path as osp
import sys
import unittest

from torch import Tensor
from unittest import TestCase

from mlu.datasets.fsd50k import FSD50KDataset
from mlu.nn import Squeeze, MultiHot


class TestFSD50K(TestCase):
	def test_item(self):
		logging.basicConfig(stream=sys.stdout, level=logging.INFO)
		dataset = FSD50KDataset(
			root=osp.join('..', 'data', 'FSD50K'),
			subset='eval',
			transform=Squeeze(),
			target_transform=MultiHot(200),
			download=False,
			verbose=2,
		)

		item = dataset[0]
		self.assertEqual(len(item), 2)
		audio, target = item
		self.assertTrue(isinstance(audio, Tensor))
		print(audio.shape)
		print(target.shape)


if __name__ == '__main__':
	unittest.main()
