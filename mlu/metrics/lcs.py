
import torch

from mlu.metrics.base import Metric
from torch import Tensor


class LongestCommonSubsequence(Metric[Tensor, Tensor, Tensor]):
	""" TODO : test """
	def compute_score(self, input_: Tensor, target: Tensor) -> Tensor:
		lengths = torch.zeros((len(input_), len(target)))

		for i in range(1, len(input_) + 1):
			for j in range(1, len(target) + 1):
				if input_[i - 1] == target[j - 1]:
					lengths[i, j] = lengths[i - 1, j - 1] + 1
				else:
					lengths[i, j] = max(lengths[i, j - 1], lengths[i - 1, j])

		return lengths[-1, -1]
