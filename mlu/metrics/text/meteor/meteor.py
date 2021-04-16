
import torch

from torch import Tensor

from mlu.metrics.base import Metric
from nltk.translate.meteor_score import meteor_score
from typing import List


class METEOR(Metric):
	def __init__(self, alpha: float = 0.9, gamma: float = 0.5, beta: float = 3.0):
		"""
			Metric for Evaluation of Translation with Explicit ORdering (METEOR)
			Use 'nltk' package as backend.

			:param alpha: Parameter for controlling the weights of precision and recall. (default: 0.9)
			:param gamma: The coefficient used in the brevity penalty function. (default: 0.5)
			:param beta: The power used in the brevity penalty function. (default: 3.0)
		"""
		super().__init__()
		self.alpha = alpha
		self.gamma = gamma
		self.beta = beta

	def compute_score(self, candidate: str, references: List[str]) -> Tensor:
		score = meteor_score(references, candidate, alpha=self.alpha, beta=self.beta, gamma=self.gamma)
		return torch.scalar_tensor(score)
