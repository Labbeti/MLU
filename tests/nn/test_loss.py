
import torch
import unittest

from unittest import TestCase
from mlu.nn import BCELossBatchMean


class TestLoss(TestCase):
	def test(self):
		pred = torch.rand(32, 10)
		target = torch.ones(32, 10)

		bce = BCELossBatchMean()

		loss = bce(pred, target)

		self.assertEqual(loss.shape, torch.Size([32]))


if __name__ == "__main__":
	unittest.main()
