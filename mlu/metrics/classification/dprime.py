
import torch

from mlu.metrics.base import Metric
from scipy.stats import norm
from sklearn.metrics import roc_auc_score
from torch import Tensor
from typing import Callable, Optional


class DPrime(Metric):
	def __init__(self, average: Optional[str] = None, reduce_fn: Callable = torch.mean):
		"""
			DPrime metric.

			Note: If score == 0 : bad score, low difference between "noise" and inputs.

			Backend: sklearn and scipy.

			:param average: The type of D' score to compute. (default: 'macro')
			:param reduce_fn: The reduction function to apply. (default: torch.mean)
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
		assert len(input_.shape) == 2

		input_ = input_.cpu().numpy()
		target = target.cpu().numpy()

		roc_auc = roc_auc_score(y_true=target, y_score=input_, average=self.average)
		score = (2 ** 0.5) * norm.ppf(roc_auc)
		score = torch.as_tensor(score)
		score = self.reduce_fn(score)

		return score


def d_prime_from_auc(auc: float) -> float:
	result = norm.ppf(auc) * 2 ** 0.5
	return result
