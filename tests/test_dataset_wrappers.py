
import torch
import unittest

from unittest import TestCase

from mlu.datasets.dummy import DummyDataset
from mlu.datasets.wrappers import NoLabelDataset


class TestDatasetWrappers(TestCase):
	def test_no_label(self):
		dataset_raw = DummyDataset()
		dataset = NoLabelDataset(dataset_raw)

		self.assertEqual(len(dataset_raw), len(dataset))
		for idx in range(len(dataset)):
			self.assertTrue(torch.allclose(dataset[idx], dataset_raw[idx][0]))

	def test_no_label_2(self):
		dataset = DummyDataset()
		dataset_w = NoLabelDataset(dataset)

		self.assertTrue(isinstance(dataset_w[0], Tensor))
		self.assertEqual(list(dataset_w[0].shape), dataset.data_shape)


if __name__ == "__main__":
	unittest.main()
