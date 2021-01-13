
import torch

from mlu.metrics.base import Metric
from scipy.stats import norm
from sklearn.metrics import roc_auc_score
from torch import Tensor
from typing import Callable


class DPrime(Metric):
	def __init__(self, average: str = "macro", reduce_fn: Callable = torch.mean):
		"""
			DPrime metric.

			Note: If score == 0 : bad score, low difference between "noise" and inputs.

			Backend: sklearn and scipy.

			:param average: The type of D' score to compute. (default: \"macro\")
			:param reduce_fn: The reduction function to apply.
		"""
		super().__init__()
		self.average = average
		self.reduce_fn = reduce_fn

	def compute_score(self, input_: Tensor, target: Tensor) -> Tensor:
		"""
			Compute DPrime score on prediction and target.

			:param input_: (nb_classes) or (batch_size, nb_classes) tensor
			:param target: (nb_classes) or (batch_size, nb_classes) tensor
			:return: The DPrime score as scalar tensor.
		"""
		assert input_.shape == target.shape
		assert 1 <= len(input_.shape) <= 2

		if len(input_.shape) == 1:
			score = roc_auc_score(y_score=input_.numpy(), y_true=target.numpy(), average=self.average)
			score = (2 ** 0.5) * norm.ppf(score)
			score = torch.as_tensor(score)
		elif len(input_.shape) == 2:
			score = [self.compute_score(input_[i], target[i]) for i in range(input_.shape[0])]
			score = torch.as_tensor(score)
			score = self.reduce_fn(score)
		else:
			raise RuntimeError(f"Invalid tensor dimension {input_.shape} for ROC AUC score. Only 1D or 2D-tensors are supported.")

		return score
