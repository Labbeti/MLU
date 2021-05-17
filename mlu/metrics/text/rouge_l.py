
from rouge_metric import PyRouge
from torch.nn import Module
from typing import List


class RougeL(Module):
	def __init__(self):
		"""
			Recall Oriented Understudy of Gisting Evaluation.
			Use 'rouge-metric' package as backend.
		"""
		super().__init__()
		self.rouge = PyRouge(rouge_l=True)

	def forward(self, hypothesis: List[List[str]], references: List[List[List[str]]]) -> float:
		if len(hypothesis) != len(references):
			raise ValueError(f'Batch size of hypothesis and references are different ({len(hypothesis)} != {len(references)}).')

		hypothesis = [' '.join(hyp) for hyp in hypothesis]
		references = [[' '.join(ref) for ref in refs] for refs in references]

		scores = self.rouge.evaluate(hypotheses=hypothesis, multi_references=references)
		rouge_l_scores = scores['rouge-l']
		# 3 scores = Recall r, Precision p, FScore f
		# {'r': ..., 'p': ..., 'f': ...}
		f_score = rouge_l_scores['f']

		return f_score
