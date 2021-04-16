
import torch
import unittest

from torch.nn import BCELoss
from unittest import TestCase

from mlu.metrics import BCEMetric


class TestBCE(TestCase):
	def test_bce(self):
		bce1 = BCELoss()
		bce2 = BCEMetric()

		pred = torch.rand(10, 5)
		target = torch.rand(10, 5).ge(0.5).float()

		l1 = bce1(pred, target)
		l2 = bce2(pred, target)
		self.assertEqual(l1, l2)


if __name__ == '__main__':
	unittest.main()
