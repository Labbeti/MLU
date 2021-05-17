
from torch.nn import Module
from typing import List

from .cider import Cider
from .spice import Spice


class Spider(Module):
	def __init__(self, cider_weight: float = 0.5, spice_weight: float = 0.5):
		super().__init__()
		self.cider_weight = cider_weight
		self.spice_weight = spice_weight

		self.cider = Cider(ngrams_max=4)
		self.spice = Spice()

	def forward(self, hypothesis: List[List[str]], references: List[List[List[str]]]) -> float:
		score_cider = self.cider(hypothesis, references)
		score_spice = self.spice(hypothesis, references)
		score = self.cider_weight * score_cider + self.spice_weight * score_spice
		return score


class SpiderFromObj(Module):
	def __init__(self, cider: Cider, spice: Spice, cider_weight: float = 0.5, spice_weight: float = 0.5):
		super().__init__()
		self.cider = cider
		self.spice = spice
		self.cider_weight = cider_weight
		self.spice_weight = spice_weight

	def forward(self, hypothesis: List[List[str]], references: List[List[List[str]]]) -> float:
		if len(hypothesis) != len(references):
			raise ValueError(f'Batch size of hypothesis and references are different ({len(hypothesis)} != {len(references)}).')

		score_cider = self.cider.get_last_score()
		score_spice = self.spice.get_last_score()
		score = self.cider_weight * score_cider + self.spice_weight * score_spice
		return score
