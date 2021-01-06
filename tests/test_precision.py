
import torch

from mlu.metrics.precision import Precision
from mlu.metrics.recall import Recall
from unittest import TestCase, main


class Precision2:
	def __init__(self, dim=None, epsilon=1e-10):
		self.dim = dim
		self.epsilon = epsilon

	def __call__(self, y_pred, y_true):
		dim = () if self.dim is None else self.dim

		true_positives = torch.sum(torch.round(torch.clamp(y_true * y_pred, 0, 1)), dim=dim)
		predicted_positives = torch.sum(torch.round(torch.clamp(y_pred, 0, 1)), dim=dim)

		if self.dim is None and predicted_positives == 0:
			self.value_ = torch.as_tensor(0.0)
		else:
			self.value_ = true_positives / (predicted_positives + self.epsilon)

		return self.value_


class Recall2:
	def __init__(self, dim=None, epsilon=1e-10):
		self.dim = dim
		self.epsilon = epsilon

	def __call__(self, y_pred, y_true):
		dim = () if self.dim is None else self.dim

		true_positives = torch.sum(torch.round(torch.clamp(y_true * y_pred, 0.0, 1.0)), dim=dim)
		possible_positives = torch.sum(torch.clamp(y_true, 0.0, 1.0), dim=dim)

		if self.dim is None and possible_positives == 0:
			self.value_ = torch.as_tensor(0.0)
		else:
			self.value_ = true_positives / (possible_positives + self.epsilon)

		return self.value_


class TestPrecision(TestCase):
	def test_precision(self):
		p1 = Precision()
		p2 = Precision2()

		input_ = torch.as_tensor([
			[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1]
		]).float()
		target = torch.as_tensor([
			[1, 0, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0]
		]).float()

		s1 = p1(input_, target)
		s2 = p2(input_, target)

		print("s1", s1)
		print("s2", s2)
		self.assertAlmostEqual(s1, s2)

	def test_recall(self):
		p1 = Recall()
		p2 = Recall2()

		input_ = torch.as_tensor([
			[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0]
		]).float()
		target = torch.as_tensor([
			[1, 0, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0]
		]).float()

		s1 = p1(input_, target)
		s2 = p2(input_, target)

		print("s1", s1)
		print("s2", s2)
		self.assertAlmostEqual(s1, s2)


if __name__ == "__main__":
	main()
