
import numpy as np
import torch
import unittest

from sklearn import metrics
from unittest import TestCase
from mlu.metrics import AveragePrecision


class Metrics:
	def __init__(self, epsilon=1e-10):
		self.values = []
		self.accumulate_value = 0
		self.epsilon = epsilon

	def reset(self):
		self.values = []

	def __call__(self, y_pred, y_true):
		pass

	@property
	def value(self):
		return self.values[-1]

	def mean(self, size: int = None):
		if size is None:
			nb_value = len(self.values)
			accumulate = sum(self.values)

		else:
			nb_value = size
			accumulate = sum(self.values[-size:])

		avg = accumulate / nb_value
		return avg

	def std(self, size: int = None):
		if size is None:
			std_ = np.std(self.values)

		else:
			std_ = np.std(self.values[-size:])

		return std_


class MAP(Metrics):
	def __init__(self, epsilon=1e-10):
		super().__init__(epsilon)

	def __call__(self, y_pred, y_true):
		super().__call__(y_pred, y_true)

		with torch.set_grad_enabled(False):
			if y_pred.is_cuda:
				y_pred = y_pred.cpu()
			if y_true.is_cuda:
				y_true = y_true.cpu()

			aps = metrics.average_precision_score(y_true, y_pred, average=None)
			aps = np.nan_to_num(aps)

			self.values.append(aps.mean())
			return self


class TestMAP(TestCase):
	def test_same(self):
		map1 = AveragePrecision()
		map2 = MAP()

		pred = torch.rand(32, 527)
		target = torch.rand(32, 527).ge(0.75).float()

		score1 = map1(pred, target)
		score2 = map2(pred, target).mean()

		score1, score2 = map(torch.as_tensor, (score1, score2))

		self.assertTrue(torch.allclose(score1, score2), f'{score1} != {score2}')


if __name__ == "__main__":
	unittest.main()
