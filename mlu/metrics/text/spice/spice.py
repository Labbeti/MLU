
from torch.nn import Module
from typing import List

from .spice_coco import Spice as SpiceCoco


class Spice(Module):
	def __init__(self):
		"""
			SPICE metric.

			Original paper : https://arxiv.org/pdf/1607.08822.pdf
		"""
		super().__init__()
		self._spice_coco = SpiceCoco()
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

		average_score, _scores = self._spice_coco.compute_score(gts, res)
		self._prev_avg_score = average_score
		return average_score

	def get_last_score(self) -> float:
		return self._prev_avg_score
