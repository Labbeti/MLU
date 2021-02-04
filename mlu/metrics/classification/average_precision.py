
import torch

from mlu.metrics.base import Metric
from sklearn.metrics import average_precision_score
from torch import Tensor
from typing import Callable, Optional


class AveragePrecision(Metric):
	def __init__(self, average: Optional[str] = None, reduce_fn: Callable = torch.mean):
		"""
			Compute mean Average Precision (mAP) score.
			Backend: scikit-learn.

			:param reduce_fn: The reduction function to apply.
		"""
		super().__init__()
		self.average = average
		self.reduce_fn = reduce_fn

	def compute_score(self, input_: Tensor, target: Tensor) -> Tensor:
		"""
			Compute mAP score on prediction and target.

			:param input_: (nb_samples, nb_classes) tensor
			:param target: (nb_samples, nb_classes) binary tensor
			:return: The mAP score as scalar tensor.
		"""
		assert input_.shape == target.shape
		assert len(input_.shape) == 2

		input_ = input_.cpu().numpy()
		target = target.cpu().numpy()

		score = average_precision_score(y_true=target, y_score=input_, average=self.average)
		score = torch.as_tensor(score)
		score = self.reduce_fn(score)

		return score
