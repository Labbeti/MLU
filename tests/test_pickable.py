
import pickle
import unittest
from unittest import TestCase

from mlu.metrics import *


class TestPickable(TestCase):
	def test_metrics(self):
		metrics = [
			AveragePrecision(),
			BinaryAccuracy(),
			CategoricalAccuracy(),
			DPrime(),
			EqMetric(),
			FScore(),
			Precision(),
			Recall(),
			RocAuc(),
			UAR(),
			BLEU(),
			LCS(),
			NIST(),
			RougeL(),
			SPIDER(),
			WordErrorRate(),
			CIDER(),
			METEOR(),
			SPICE(),
		]

		try:
			for metric in metrics:
				tmp = pickle.dumps(metric)
				self.assertTrue(isinstance(tmp, bytes))
				self.assertGreater(len(tmp), 0)
		except AttributeError:
			self.assertTrue(False)


if __name__ == "__main__":
	unittest.main()
