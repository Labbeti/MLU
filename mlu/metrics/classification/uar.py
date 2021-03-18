
import torch

from sklearn.metrics import recall_score
from torch import Tensor
from typing import Callable

from mlu.metrics.base import Metric


class UAR(Metric):
	def __init__(self, vector_input: bool = True, vector_target: bool = True, reduce_fn: Callable = torch.mean):
		super().__init__()
		self.vector_input = vector_input
		self.vector_target = vector_target
		self.reduce_fn = reduce_fn
		self.average = "macro"

	def compute_score(self, input_: Tensor, target: Tensor) -> Tensor:
		assert input_.shape == target.shape
		assert len(input_.shape) == 2

		if self.vector_input:
			input_ = input_.argmax(dim=-1)

		if self.vector_target:
			target = target.argmax(dim=-1)

		input_ = input_.cpu().numpy()
		target = target.cpu().numpy()

		score = recall_score(y_true=target, y_pred=input_, average=self.average)
		score = torch.as_tensor(score)
		score = self.reduce_fn(score)

		return score
