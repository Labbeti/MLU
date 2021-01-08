
import torch

from mlu.metrics.base import Metric
from torch import Tensor
from typing import Sequence


class LCS(Metric):
	"""
		Longest Common Subsequence (LCS) metric.
		TODO : test
	"""
	def compute_score(self, input_: Sequence, target: Sequence) -> Tensor:
		assert len(input_) > 0 and len(target) > 0
		lengths = torch.zeros((len(input_), len(target)))

		for i in range(1, len(input_) + 1):
			for j in range(1, len(target) + 1):
				if input_[i - 1] == target[j - 1]:
					lengths[i, j] = lengths[i - 1, j - 1] + 1
				else:
					lengths[i, j] = max(lengths[i, j - 1], lengths[i - 1, j])

		return lengths[-1, -1]
