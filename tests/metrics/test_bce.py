
import torch
import unittest

from torch.nn import BCELoss
from unittest import TestCase

from mlu.metrics.base import Metric


class BCEMetric(Metric):
	def __init__(self):
		super().__init__()
		self.reduce_fn = torch.mean

	def compute_score(self, pred, target):
		if pred.shape != target.shape:
			raise RuntimeError(f"Mismatch shapes {pred.shape} != {target.shape}.")

		scores = - target * torch.log(pred) - (1.0 - target) * torch.log(1.0 - pred)
		scores = self.reduce_fn(scores)
		return scores


class TestBCE(TestCase):
	def test_bce(self):
		bce1 = BCELoss()
		bce2 = BCEMetric()

		pred = torch.rand(10, 5)
		target = torch.rand(10, 5).ge(0.5).float()

		l1 = bce1(pred, target)
		l2 = bce2(pred, target)
		self.assertEqual(l1, l2)


if __name__ == "__main__":
	unittest.main()
