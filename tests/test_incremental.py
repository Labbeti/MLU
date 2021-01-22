
import random
import torch

from mlu.metrics import IncrementalMean, IncrementalStd, MinTracker, MaxTracker
from unittest import TestCase, main


class TestIncremental(TestCase):
	PLACES = 5

	def test_incremental_mean(self):
		inc = IncrementalMean()

		test = torch.rand([20])
		for v in test:
			inc.add(v)

		self.assertAlmostEqual(inc.get_current().item(), test.mean().item(), places=self.PLACES)

	def test_incremental_std(self):
		unbiased = False
		inc = IncrementalStd(unbiased=unbiased)

		test = torch.rand([20])
		for v in test:
			inc.add(v)

		self.assertAlmostEqual(inc.get_std().item(), test.std(unbiased=unbiased).item(), places=self.PLACES)

	def test_trackers(self):
		min_tracker = MinTracker()
		max_tracker = MaxTracker()

		values = list(range(100))
		random.shuffle(values)

		for v in values:
			min_tracker.add(v)
			max_tracker.add(v)

		self.assertEqual(min_tracker.get_current(), 0)
		self.assertEqual(max_tracker.get_current(), 99)

	def test_tracker_rand(self):
		values = torch.rand(100)
		tracker = MaxTracker()

		tracker.add_values(values)
		self.assertAlmostEqual(tracker.get_max().item(), values.max(), places=self.PLACES)


if __name__ == "__main__":
	main()
