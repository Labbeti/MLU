
import torch

from sklearn.metrics import average_precision_score
from torch import Tensor
from typing import Callable, Optional

from mlu.metrics.base import Metric


class AveragePrecision(Metric):
	def __init__(
		self,
		average: Optional[str] = None,
		reduce_fn: Callable = torch.mean,
	):
		"""
			Compute mean Average Precision (mAP) score.

			Backend: scikit-learn, see https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html.

			:param average: The average mode used by sklearn function 'average_precision_score'.
				Can be None, 'micro', 'macro', 'weighted' or 'samples'. (default: None)
			:param reduce_fn: The reduction function to apply. (default: torch.mean)
		"""
		super().__init__()
		self.average = average
		self.reduce_fn = reduce_fn

	def compute_score(self, input_: Tensor, target: Tensor) -> Tensor:
		"""
			Compute mAP score on prediction and target.

			:param input_: (nb_samples, num_classes) tensor
			:param target: (nb_samples, num_classes) binary tensor
			:return: The mAP score as scalar tensor.
		"""
		assert input_.shape == target.shape
		assert len(input_.shape) == 2
		assert target.eq(0.0).logical_or(target.eq(1.0)).all(), "Target must be binary tensor containing only 0 and 1."

		input_ = input_.cpu().numpy()
		target = target.cpu().numpy()

		score = average_precision_score(y_true=target, y_score=input_, average=self.average)
		score = torch.as_tensor(score)
		score = self.reduce_fn(score)

		return score
