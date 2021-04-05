
import pickle
import unittest
from unittest import TestCase

from mlu.metrics import *
from mlu.nn import *


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

	def test_nn(self):
		modules = [
			CrossEntropyWithVectors(),
			KLDivLossWithProbabilities(),
			KLDivLoss(),
			JSDivLoss(),
			JSDivLossFromLogits(),
			Entropy(),
			Mean(),
			Max(),
			Min(),
		]

		try:
			for m in modules:
				tmp = pickle.dumps(m)
				self.assertTrue(isinstance(tmp, bytes))
				self.assertGreater(len(tmp), 0)
		except AttributeError:
			self.assertTrue(False)


if __name__ == "__main__":
	unittest.main()
