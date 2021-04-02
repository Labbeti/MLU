
import torch
import unittest

from torch.nn.functional import one_hot
from torch.distributions.categorical import Categorical
from unittest import TestCase

from mlu.metrics import CategoricalAccuracy, UAR


class TestDatasetWrappers(TestCase):
	def test_no_label(self):
		# pred = torch.rand(1000, 5)
		# pred = pred / pred.norm(p=1, dim=1, keepdim=True)
		#
		# target = (torch.rand(pred.shape[0]) * pred.shape[1]).floor().long()
		# target = one_hot(target, pred.shape[1])

		bsize = 1000

		pred = torch.zeros(bsize)
		probs = torch.as_tensor([0.2, 0.2, 0.2, 0.2, 0.2])
		law = Categorical(probs=probs / probs.norm(1, 0, True))
		pred = law.sample((bsize,))

		probs = torch.as_tensor([0.8, 0.05, 0.05, 0.05])
		law = Categorical(probs=probs / probs.norm(1, 0, True))
		target = law.sample((bsize,))

		indices = target != 0
		pred[indices] = target[indices]

		print(pred)
		print(target)

		kwargs = dict(vector_input=False, vector_target=False)
		acc = CategoricalAccuracy(**kwargs)
		uar = UAR(**kwargs)

		score = acc(pred, target)
		print("Acc: ", score)

		score = uar(pred, target)
		print("UAR: ", score)


if __name__ == "__main__":
	unittest.main()
