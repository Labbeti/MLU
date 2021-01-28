
from mlu.metrics.text.cider.cider_coco import Cider
from mlu.metrics.base import Metric
from typing import List


class CIDER(Metric):
	def __init__(self, ngrams_max: int = 4):
		"""
			Consensus-based Image Description Evaluation.

			Original paper : http://arxiv.org/pdf/1411.5726.pdf

			:param ngrams_max:
		"""
		super().__init__()
		self._cider_internal = Cider(n=ngrams_max)

	def compute_score(self, reference: List[str], hypothesis: List[List[str]]) -> float:
		gts = {"no_id": reference}
		res = {"no_id": hypothesis}
		average_score, scores = self._cider_internal.compute_score(gts, res)
		return average_score
