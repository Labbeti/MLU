
from mlu.metrics.base import Metric
from torch import Tensor


class Recall(Metric):
	"""
		Compute Recall score between binary vectors.
		Recall = TP / (TP + FN) where TP = True Positives, FN = False Negatives

		Vectors must be 1D-tensors of shape (nb classes)
	"""
	def compute_score(self, input_: Tensor, target: Tensor) -> Tensor:
		"""
			Compute score with one-hot or multi-hot inputs and targets.

			:param input_: Shape (nb classes)
			:param target: Shape (nb classes)
			:return: Shape (1,)
		"""
		assert input_.shape == target.shape, \
			f"Mismatch between shapes {str(input_.shape)} and {str(target.shape)} for Recall metric."
		true_positives = (input_ * target).sum()
		false_negatives = (target - input_).ge(1.0).sum()
		return true_positives / (true_positives + false_negatives)
