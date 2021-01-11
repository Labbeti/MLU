
import torch

from mlu.metrics.base import Metric
from rouge_metric import PyRouge
from torch import Tensor
from typing import List


class RougeL(Metric[List[str], List[List[str]], Tensor]):
	def __init__(self):
		"""
			Recall Oriented Understudy of Gisting Evaluation.
			Use "rouge-metric" package as backend.
		"""
		super().__init__()
		self.rouge = PyRouge(rouge_l=True)

	def compute_score(self, references: List[str], hypothesis: List[List[str]]) -> Tensor:
		scores = self.rouge.evaluate(references, hypothesis)
		rouge_l_scores = scores["rouge-l"]
		# 3 scores = Recall r, Precision p, FScore f
		# {"r": ..., "p": ..., "f": ...}
		f_score = rouge_l_scores["f"]
		return torch.scalar_tensor(f_score)
