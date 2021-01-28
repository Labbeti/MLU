
from mlu.metrics.text.cider.cider import CIDER
from mlu.metrics.text.spice.spice import SPICE
from mlu.metrics.base import Metric
from typing import List


class SPIDER(Metric):
	def __init__(self, weight_cider: float = 0.5, weight_spice: float = 0.5):
		super().__init__()
		self.weight_cider = weight_cider
		self.weight_spice = weight_spice

		self.cider = CIDER(ngrams_max=4)
		self.spice = SPICE()

	def compute_score(self, reference: List[str], hypothesis: List[List[str]]) -> float:
		score_cider = self.cider(reference, hypothesis)
		score_spice = self.spice(reference, hypothesis)
		score = self.weight_cider * score_cider + self.weight_spice * score_spice
		return score
