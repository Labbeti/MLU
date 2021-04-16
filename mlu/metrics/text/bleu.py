
from torch.nn import Module
from torchtext.data.metrics import bleu_score
from typing import List


class Bleu(Module):
	def __init__(self, max_n: int = 4):
		super().__init__()
		self.max_n = max_n
		self.weights = [1.0 / max_n for _ in range(max_n)]

	def forward(self, hypothesis: List[List[any]], references: List[List[List[any]]]) -> float:
		if len(hypothesis) != len(references):
			raise ValueError(f'Batch size of hypothesis and references are different ({len(hypothesis)} != {len(references)}).')

		score = bleu_score(hypothesis, references, self.max_n, self.weights)
		return score
