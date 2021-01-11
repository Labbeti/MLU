
import torch

from mlu.metrics import IncrementalMean, IncrementalStd
from unittest import TestCase, main


class TestIncremental(TestCase):
	def test_incremental_mean(self):
		inc = IncrementalMean()

		test = torch.rand([20])
		for v in test:
			inc.add(v)

		self.assertEqual(inc.get_mean(), test.mean())


if __name__ == "__main__":
	main()
