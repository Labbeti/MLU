
import numpy as np
import torch

from mlu.nn import OneHot, MultiHot
from unittest import TestCase, main


class TestOneHot(TestCase):
	def test_1(self):
		to_onehot = OneHot(n_classes=5)

		indexes = [2, np.array([2]), torch.scalar_tensor(2, dtype=torch.long)]
		expected_lst = [torch.as_tensor([0, 0, 1, 0, 0], dtype=torch.float)] * len(indexes)
		for idx, expected in zip(indexes, expected_lst):
			one_hot = to_onehot(idx)
			if isinstance(one_hot, np.ndarray):
				one_hot = torch.from_numpy(one_hot)
			self.assertTrue(one_hot.eq(expected).all())

	def test_2(self):
		onehot = OneHot(n_classes=5)
		idx = 1

		r = onehot(idx)
		self.assertTrue(torch.eq(r, torch.as_tensor([0, 1, 0, 0, 0])).all())


class TestMultiHot(TestCase):
	def test_1(self):
		multihot = MultiHot(n_classes=3)
		targets = [[1, 2, 0], [1], [], [2, 2]]
		expected = [[1, 1, 1], [0, 1, 0], [0, 0, 0], [0, 0, 1]]

		expected = torch.as_tensor(expected)
		result = multihot(targets)
		self.assertTrue(result.eq(expected).all())

	def test_2(self):
		multihot = MultiHot(n_classes=3)
		targets = 2
		expected = [0, 0, 1]

		expected = torch.as_tensor(expected)
		result = multihot(targets)
		self.assertTrue(result.eq(expected).all(), f'Diff : {result} != {expected}')


if __name__ == '__main__':
	main()
