import unittest

from mlu.datasets.utils import split_indexes_per_class, flat_split_indexes_per_class
from unittest import TestCase


class TestSplitIdx(TestCase):
	def test_flat_expected(self):
		tests = [
			([[1, 2], [3, 4], [5, 6]], [0.5, 0.5]),
			([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], [0.5, 0.25, 0.25]),
		]
		expected_lst = [
			[[1, 3, 5], [2, 4, 6]],
			[[1, 2, 5, 6, 9, 10], [3, 7, 11], [4, 8, 12]],
		]
		for (indices, ratios), expected in zip(tests, expected_lst):
			splits = split_indexes_per_class(indices, ratios)
			splits = flat_split_indexes_per_class(splits)
			self.assertEqual(splits, expected)

	def test_flat_reduce(self):
		cls_idx = [[0, 1, 2, 3], [4, 5, 6, 7]]
		expected = [[0, 1, 4, 5]]
		ratios = [0.5]

		splits = split_indexes_per_class(cls_idx, ratios)
		splits = flat_split_indexes_per_class(splits)
		self.assertEqual(splits, expected)

	def test_flat_overlap(self):
		indexes_per_class = [
			list(range(0, 10)),
			list(range(10, 20)),
			list(range(20, 30)),
		]
		ratios = [0.1, 0.2]

		splits = split_indexes_per_class(indexes_per_class, ratios)
		splits = flat_split_indexes_per_class(splits)
		self.assertEqual(splits, [[0, 10, 20], [1, 2, 11, 12, 21, 22]])

	def test_expected_2(self):
		tests_params = [
			dict(
				indexes_per_class=[[1, 2], [3, 4], [5, 6]],
				ratios=[0.5, 0.5],
			),
			dict(
				indexes_per_class=[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
				ratios=[0.5, 0.25, 0.25],
			),
			dict(
				indexes_per_class=[],
				ratios=[],
			),
			dict(
				indexes_per_class=[[1, 2, 3]],
				ratios=[0.5, 0.5],
			),
			dict(
				indexes_per_class=[[1], []],
				ratios=[1.0],
			),
		]

		tests_expected_returns = [
			[[[1], [3], [5]], [[2], [4], [6]]],
			[[[1, 2], [5, 6], [9, 10]], [[3], [7], [11]], [[4], [8], [12]]],
			[],
			[[[1, 2]], [[3]]],
			[[[1], []]],
		]

		for params, expected_return in zip(tests_params, tests_expected_returns):
			split = split_indexes_per_class(**params)
			self.assertEqual(split, expected_return)


if __name__ == '__main__':
	unittest.main()
