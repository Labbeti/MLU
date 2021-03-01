import unittest

from mlu.datasets.utils import split_indexes_per_class_flat, split_indexes_per_class
from unittest import TestCase


class TestSplitIdxFlat(TestCase):
	def test_expected(self):
		tests = [
			([[1, 2], [3, 4], [5, 6]], [0.5, 0.5]),
			([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], [0.5, 0.25, 0.25]),
		]
		expected_lst = [
			[[1, 3, 5], [2, 4, 6]],
			[[1, 2, 5, 6, 9, 10], [3, 7, 11], [4, 8, 12]],
		]
		for (indices, ratios), expected in zip(tests, expected_lst):
			idx_split = split_indexes_per_class_flat(indices, ratios)
			self.assertEqual(idx_split, expected)

	def test_reduce(self):
		cls_idx = [[0, 1, 2, 3], [4, 5, 6, 7]]
		expected = [0, 1, 4, 5]
		idx_split = split_indexes_per_class_flat(cls_idx, [0.5])[0]
		self.assertEqual(idx_split, expected)

	def test_overlap(self):
		indexes_per_class = [
			list(range(0, 10)),
			list(range(10, 20)),
			list(range(20, 30)),
		]
		ratios = [0.1, 0.2]
		split = split_indexes_per_class_flat(indexes_per_class, ratios)

		self.assertEqual(split, [[0, 10, 20], [1, 2, 11, 12, 21, 22]])


class TestSplitIdx(TestCase):
	def test_expected(self):
		tests = [
			dict(
				indexes_per_class=[[1, 2], [3, 4], [5, 6]],
				ratios=[0.5, 0.5]),
			dict(
				indexes_per_class=[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
				ratios=[0.5, 0.25, 0.25]),
			dict(
				indexes_per_class=[],
				ratios=[]),
			dict(
				indexes_per_class=[[1, 2, 3]],
				ratios=[0.5, 0.5]),
			dict(
				indexes_per_class=[[1], []],
				ratios=[1.0]),
		]
		expected_splits = [
			[[[1], [3], [5]], [[2], [4], [6]]],
			[[[1, 2], [5, 6], [9, 10]], [[3], [7], [11]], [[4], [8], [12]]],
			[],
			[[[1, 2]], [[3]]],
			[[[1], []]]
		]
		for test, expected in zip(tests, expected_splits):
			split = split_indexes_per_class(**test)
			self.assertEqual(split, expected)


if __name__ == '__main__':
	unittest.main()
