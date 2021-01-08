
import torch
from mlu.metrics.base import Metric
from torch import Tensor
from typing import Callable


class Recall(Metric):
	"""
		Compute Recall score between binary vectors.
		Recall = TP / (TP + FN) where TP = True Positives, FN = False Negatives

		Vectors must be 1D-tensors of shape (nb classes)
	"""
	def __init__(self, dim: int = 1, reduce_fn: Callable = torch.mean):
		super().__init__()
		self.dim = dim
		self.reduce_fn = reduce_fn

	def compute_score(self, input_: Tensor, target: Tensor) -> Tensor:
		"""
			Compute score with one-hot or multi-hot inputs and targets.

			:param input_: Shape (nb classes)
			:param target: Shape (nb classes)
			:return: Shape (1,)
		"""
		assert input_.shape == target.shape, \
			f"Mismatch between shapes {str(input_.shape)} and {str(target.shape)} for Recall metric."
		true_positives = (input_ * target).sum(dim=self.dim)
		false_negatives = (target - input_).ge(1.0).sum(dim=self.dim)
		score = true_positives / (true_positives + false_negatives)
		score = self.reduce_fn(score)
		return score
