
import torch

from mlu.metrics.base import Metric
from sklearn.metrics import roc_auc_score
from torch import Tensor
from typing import Callable, Optional


class RocAuc(Metric):
	def __init__(self, average: Optional[str] = None, reduce_fn: Callable = torch.mean):
		"""
			Compute mean Area Under the Receiver Operating Characteristic Curve (ROC AUC) score.
			Backend: scikit-learn

			:param average: The type of ROC AUC score to compute. (default: 'macro')
			:param reduce_fn: The reduction function to apply.
		"""
		super().__init__()
		self.average = average
		self.reduce_fn = reduce_fn

	def compute_score(self, input_: Tensor, target: Tensor) -> Tensor:
		"""
			Compute ROC AUC score on prediction and target.

			:param input_: (batch_size, nb_classes) tensor
			:param target: (batch_size, nb_classes) tensor
			:return: The ROC AUC score as scalar tensor.
		"""
		assert input_.shape == target.shape
		assert len(input_.shape) == 2

		input_ = input_.cpu().numpy()
		target = target.cpu().numpy()

		score = roc_auc_score(y_true=target, y_score=input_, average=self.average)
		score = torch.as_tensor(score)
		score = self.reduce_fn(score)

		return score
