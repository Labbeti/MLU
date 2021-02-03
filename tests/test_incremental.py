
import random
import torch

from mlu.metrics.incremental import IncrementalMean, IncrementalStd, MinTracker, MaxTracker, NBestsTracker
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

		min_tracker.reset()
		self.assertIsNone(min_tracker.get_current())
		self.assertTrue(min_tracker.is_empty())

		max_tracker.reset()
		self.assertIsNone(max_tracker.get_current())
		self.assertTrue(max_tracker.is_empty())

	def test_tracker_rand(self):
		values = torch.rand(100)
		tracker = MaxTracker()

		tracker.add_values(values)
		self.assertAlmostEqual(tracker.get_max().item(), values.max(), places=self.PLACES)

		tracker.reset()
		self.assertIsNone(tracker.get_current())
		self.assertTrue(tracker.is_empty())

	def test_tracker_min_rand(self):
		values = torch.rand(100)
		tracker = MinTracker(*values)
		self.assertAlmostEqual(tracker.get_min().item(), values.min(), places=self.PLACES)

		tracker.reset()
		self.assertIsNone(tracker.get_current())
		self.assertTrue(tracker.is_empty())

	def test_n_bests_tracker(self):
		tracker = NBestsTracker(n=3, is_better=lambda x, y: x > y)

		values = [5, 2, 7, 6, 7, 2]
		tracker.add_values(values)

		self.assertTrue(tracker.get_current().eq(torch.as_tensor([7, 7, 6])).all())
		self.assertTrue(tracker.get_index() == [2, 4, 3])

		tracker.reset()
		self.assertIsNone(tracker.get_current())
		self.assertTrue(tracker.is_empty())


if __name__ == "__main__":
	main()
