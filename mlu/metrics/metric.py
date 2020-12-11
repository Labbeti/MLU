
from mlu.metrics.base import Metric
from torch import Tensor


class EqMetric(Metric):
	def __init__(self, dim: int):
		super().__init__()
		self.dim = dim

	def compute_score(self, input_: Tensor, target: Tensor) -> Tensor:
		return input_.eq(target).all(self.dim).float()


class Precision(Metric):
	def compute_score(self, input_: Tensor, target: Tensor) -> Tensor:
		"""
			Compute score with one-hot or multi-hot inputs and targets.

			:param input_: Shape (nb classes)
			:param target: Shape (nb classes)
			:return: Shape (1,)
		"""
		assert input_.shape == target.shape, f"Mismatch between shapes {str(input_.shape)} and {str(target.shape)}."
		true_positives = (input_ * target).sum()
		false_positives = (input_ - target).ge(1.0).sum()
		return true_positives / (true_positives + false_positives)


class Recall(Metric):
	def compute_score(self, input_: Tensor, target: Tensor) -> Tensor:
		"""
			Compute score with one-hot or multi-hot inputs and targets.

			:param input_: Shape (nb classes)
			:param target: Shape (nb classes)
			:return: Shape (1,)
		"""
		assert input_.shape == target.shape, f"Mismatch between shapes {str(input_.shape)} and {str(target.shape)}."
		true_positives = (input_ * target).sum()
		false_negatives = (target - input_).ge(1.0).sum()
		return true_positives / (true_positives + false_negatives)
