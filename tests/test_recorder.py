
import numpy as np
import unittest

from unittest import TestCase

from mlu.utils.recorder.recorder import Recorder


class TestRecorder(TestCase):
	def test_means(self):
		recorder = Recorder()

		test_set = {"a": [1, 2, 3, 4], "b": [0, -1]}
		for name, values in test_set.items():
			for v in values:
				recorder.add_scalar(name, v, 0, 0)

		epoch = 0

		for n, mean in recorder.get_current(epoch, "mean").items():
			expected = np.mean(test_set[n]).item()
			self.assertAlmostEqual(mean, expected)


if __name__ == "__main__":
	unittest.main()
