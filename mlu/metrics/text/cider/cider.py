
from torch.nn import Module
from typing import List

from .cider_coco import Cider as CiderCoco


class Cider(Module):
	def __init__(self, ngrams_max: int = 4):
		"""
			Consensus-based Image Description Evaluation.

			Original paper : https://arxiv.org/pdf/1411.5726.pdf

			:param ngrams_max: TODO
		"""
		super().__init__()
		self._cider_coco = CiderCoco(n=ngrams_max)
		self._prev_avg_score = 0.0

	def forward(self, hypothesis: List[List[str]], references: List[List[List[str]]]) -> float:
		if len(hypothesis) != len(references):
			raise ValueError(f'Batch size of hypothesis and references are different ({len(hypothesis)} != {len(references)}).')

		gts = {}
		res = {}
		for i, (hyp, refs) in enumerate(zip(hypothesis, references)):
			hyp = [' '.join(hyp)]
			refs = [' '.join(ref) for ref in refs]
			gts[i] = refs
			res[i] = hyp

		average_score, _scores = self._cider_coco.compute_score(gts, res)
		return average_score

	def get_last_score(self) -> float:
		return self._prev_avg_score
