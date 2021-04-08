
import torch
import unittest

from unittest import TestCase
from mlu.metrics import MetricDict, CategoricalAccuracy, FScore


class TestMetricDict(TestCase):
	def test(self):
		metrics = {"acc": CategoricalAccuracy(), "f1": FScore()}
		md = MetricDict(metrics)

		self.assertEqual(len(md), len(metrics))

		pred = torch.rand(32, 10)
		target = torch.ones(32, 10)

		scores = md(pred, target)
		# print(scores)
		self.assertEqual(len(scores), len(metrics))
		self.assertIn("acc", scores.keys())
		self.assertIn("f1", scores.keys())

	def test_none(self):
		in_ = None
		md = MetricDict(in_, prefix="p/", suffix="_tmp")

		self.assertEqual(len(md), 0)

		pred = torch.rand(32, 10)
		target = torch.ones(32, 10)
		scores = md(pred, target)

		self.assertEqual(scores, {})


if __name__ == "__main__":
	unittest.main()
