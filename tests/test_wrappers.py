
import unittest

from torch import Tensor
from unittest import TestCase

from mlu.datasets.dummy import DummyDataset
from mlu.datasets.wrappers import NoLabelDataset


class TestDatasetWrappers(TestCase):
	def test_no_label(self):
		dataset = DummyDataset()
		dataset_w = NoLabelDataset(dataset)

		self.assertTrue(isinstance(dataset_w[0], Tensor))
		self.assertEqual(list(dataset_w[0].shape), dataset.data_shape)


if __name__ == "__main__":
	unittest.main()
