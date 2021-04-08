
import numpy as np
import torch

from mlu.nn import OneHot
from unittest import TestCase, main


class TestOneHot(TestCase):
	def test_1(self):
		to_onehot = OneHot(num_classes=5)

		indexes = [2, np.array([2]), torch.scalar_tensor(2, dtype=torch.long)]
		expected_lst = [torch.as_tensor([0, 0, 1, 0, 0], dtype=torch.float)] * len(indexes)
		for idx, expected in zip(indexes, expected_lst):
			one_hot = to_onehot(idx)
			if isinstance(one_hot, np.ndarray):
				one_hot = torch.from_numpy(one_hot)
			self.assertTrue(one_hot.eq(expected).all())


if __name__ == "__main__":
	main()
