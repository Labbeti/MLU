
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


if __name__ == "__main__":
	unittest.main()
