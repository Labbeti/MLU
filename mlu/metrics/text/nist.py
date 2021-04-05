
from mlu.metrics.base import Metric
from nltk.translate.nist_score import sentence_nist
from torch import Tensor
from typing import List


class NIST(Metric):
	def __init__(self, ngram_order: int = 5):
		"""
			Metric for evaluate the quality of a machine translation.
			Use "nltk" package as backend.

			:param ngram_order: The ngram max order to use. (default: 5)
		"""
		super().__init__()
		self.ngram_order = ngram_order

	def compute_score(self, candidate: List[str], references: List[List[str]]) -> Tensor:
		score = sentence_nist(candidate, references, n=self.ngram_order)
		return score
