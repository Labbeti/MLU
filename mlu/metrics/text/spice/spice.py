
from mlu.metrics.text.spice.spice_coco import Spice
from mlu.metrics.base import Metric
from typing import List


class SPICE(Metric):
	def __init__(self):
		"""
			SPICE metric.

			Original paper : https://arxiv.org/pdf/1607.08822.pdf
		"""
		super().__init__()
		self._spice_internal = Spice()

	def compute_score(self, reference: List[str], hypothesis: List[List[str]]) -> float:
		gts = {"no_id": reference}
		res = {"no_id": hypothesis}
		average_score, scores = self._spice_internal.compute_score(gts, res)
		return average_score
