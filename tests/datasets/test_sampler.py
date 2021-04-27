
import unittest

from torch.utils.data.sampler import SubsetRandomSampler
from unittest import TestCase

from mlu.datasets.samplers import SubsetCycleSampler


class TestSRCSampler(TestCase):
	def test_limit(self):
		indexes = list(range(10))
		srs = SubsetRandomSampler(indexes)
		srcs = SubsetCycleSampler(indexes, 20)

		result_srs = list(srs)
		result_srcs = list(srcs)

		self.assertEqual(set(result_srs), set(indexes))
		self.assertEqual(set(result_srcs[:10]), set(indexes))
		self.assertEqual(set(result_srcs[10:]), set(indexes))

	def test_distinct(self):
		indexes = list(range(10, 20))
		n_max_iterations = 1000
		sampler = SubsetCycleSampler(indexes, n_max_iterations)

		result = list(sampler)
		splits = [result[i*len(indexes):(i+1)*len(indexes)] for i in range(n_max_iterations // len(indexes))]

		for i in range(len(splits) - 1):
			for s1, s2 in zip(splits[i:], splits[i+1:]):
				self.assertEqual(set(s1), set(s2))
				self.assertEqual(set(s1), set(indexes))

	def test_seen_every_sample(self):
		indexes = list(range(100, 200))
		sampler = SubsetCycleSampler(indexes)

		result = list(sampler)
		expected = list(indexes)

		self.assertEqual(len(result), len(expected))
		self.assertEqual(set(result), set(expected))


if __name__ == '__main__':
	unittest.main()
