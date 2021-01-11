
from mlu.metrics.base import Metric
from nltk.translate.nist_score import sentence_nist
from torch import Tensor
from typing import List


class NIST(Metric[List[str], List[List[str]], Tensor]):
	def __init__(self, ngram_order: int):
		"""
			Metric for evaluate the quality of a machine translation.
			Use "nltk" package as backend.

			:param ngram_order: The ngram max order to use.
		"""
		super().__init__()
		self.ngram_order = ngram_order

	def compute_score(self, candidate: List[str], references: List[List[str]]) -> Tensor:
		score = sentence_nist(candidate, references, n=self.ngram_order)
		return score
