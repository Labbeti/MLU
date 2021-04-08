
import torch
import unittest

from sklearn.metrics import f1_score
from unittest import TestCase

from mlu.metrics import BinaryAccuracy, FScore, Precision, Recall
from mlu.nn import Thresholding


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


class BinaryAccuracy2:
	def __init__(self, epsilon=1e-10):
		super().__init__()
		self.epsilon = epsilon

	def __call__(self, y_pred, y_true):
		y_pred = (y_pred > 0.5).float()
		correct = y_pred.eq(y_true).float().sum()
		value_ = correct / torch.prod(torch.as_tensor(y_true.shape))
		return value_


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

		# print("s1", s1)
		# print("s2", s2)
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

		# print("s1", s1)
		# print("s2", s2)
		self.assertAlmostEqual(s1, s2)

	def test_binacc(self):
		binacc = BinaryAccuracy()
		binacc2 = BinaryAccuracy2()

		pred = torch.rand(8, 10)
		target = torch.rand(8, 10).ge(0.5).float()

		score = binacc(pred, target)
		score2 = binacc2(pred, target)

		self.assertAlmostEqual(score, score2)

	def test_f1_with_sklearn(self):
		pred = torch.rand(10)
		target = torch.rand(10).ge(0.5).float()

		thres = Thresholding(0.5)
		pred = thres(pred)

		f1 = FScore()

		sk_score = f1_score(y_pred=pred.cpu().numpy(), y_true=target.cpu().numpy())
		mlu_score = f1(pred, target).item()

		self.assertAlmostEqual(sk_score, mlu_score)


class TestMetricDict(TestCase):
	def test(self):
		pred = torch.as_tensor([0, 0.9, 0.85, 0])
		target = torch.as_tensor([0, 1, 1, 0])

		binacc = BinaryAccuracy()
		f1 = FScore()

		score = binacc(pred, target)
		self.assertEqual(score, torch.ones(1))

		score = f1(pred, target)
		self.assertEqual(score, torch.ones(1))


class TestSimplify(TestCase):
	def test_recall(self):
		input_ = torch.rand(10, 1000).ge(0.5).float()
		target = torch.rand(10, 1000).ge(0.5).float()
		dim = -1

		true_positives = (input_ * target).sum(dim=dim)
		false_negatives = (target - input_).ge(1.0).sum(dim=dim)
		score = true_positives + false_negatives

		# Maybe simplify TP + FN = Possible positives = sum(target)
		score_v2 = target.sum(dim)
		self.assertTrue(torch.allclose(score, score_v2))

	def test_precision(self):
		input_ = torch.rand(10, 1000).ge(0.5).float()
		target = torch.rand(10, 1000).ge(0.5).float()
		dim = -1

		true_positives = (input_ * target).sum(dim=dim)
		false_positives = (input_ - target).ge(1.0).sum(dim=dim)
		score = true_positives + false_positives

		# Simplify TP + FP = Predicted positives = sum(input)
		score_v2 = input_.sum(dim)
		self.assertTrue(torch.allclose(score, score_v2))


if __name__ == "__main__":
	unittest.main()
