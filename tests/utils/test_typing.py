
import unittest

from torch.utils.data.dataset import Dataset
from typing import Sized
from unittest import TestCase
from mlu.datasets.dummy import DummyDataset
from mlu.utils.typing_.classes import SizedDataset


class DummyDatasetNotSized(Dataset):
	def __getitem__(self, item):
		return item


class TestTyping(TestCase):

	def test_types(self):
		dataset = DummyDataset()
		self.assertTrue(isinstance(dataset, Dataset))
		self.assertTrue(isinstance(dataset, Sized))
		self.assertTrue(isinstance(dataset, SizedDataset))

		self.pass_type_checking_1(dataset)
		self.pass_type_checking_2(dataset)
		self.pass_type_checking_3(dataset)

		dataset_2 = DummyDatasetNotSized()
		self.assertTrue(isinstance(dataset_2, Dataset))
		self.assertFalse(isinstance(dataset_2, Sized))
		self.assertFalse(isinstance(dataset_2, SizedDataset))

	def pass_type_checking_1(self, dataset: Sized):
		print("Hi1 ", len(dataset))

	def pass_type_checking_2(self, dataset: SizedDataset):
		print("Hi2 ", len(dataset))

	def pass_type_checking_3(self, dataset: Dataset):
		print("Hi3 ")


if __name__ == "__main__":
	unittest.main()
