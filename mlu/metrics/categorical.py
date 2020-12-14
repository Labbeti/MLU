
from mlu.metrics.base import Metric
from torch import Tensor


class CategoricalAccuracy(Metric):
	"""
		Compute the categorical accuracy between a batch of prediction and labels.
	"""
	def __init__(self, vector_input: bool = True, vector_target: bool = True, dim: int = 1):
		super().__init__()
		self.vector_input = vector_input
		self.vector_target = vector_target
		self.dim = dim

	def compute_score(self, input_: Tensor, target: Tensor) -> Tensor:
		if self.vector_input:
			input_ = input_.argmax(dim=self.dim)
		if self.vector_target:
			target = target.argmax(dim=self.dim)

		assert input_.shape == target.shape, "Input and target must have the same shape."
		score = input_.eq(target).float().mean()
		return score
