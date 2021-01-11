
from torch import Tensor

from mlu.metrics.translation.bleu import Metric


class WordErrorRate(Metric[Tensor, Tensor, Tensor]):
	"""
		Word Error Rate metric. The score is good if close to 0.0.
		TODO : test
	"""
	def compute_score(self, input_: Tensor, target: Tensor) -> Tensor:
		""" Compute WER """
		hist_input_ = {v: input_.eq(v).sum().item() for v in input_}
		hist_target = {v: target.eq(v).sum().item() for v in target}

		hist_correct = hist_input_
		hist_correct.update({
			key: min(hist_correct[key], hist_target[key])
			for key in hist_target.keys()
			if key in hist_correct.keys()
		})

		correct_words = sum(hist_correct.values())
		inserted_words = sum(hist_input_.values()) - correct_words
		return (correct_words - inserted_words) / len(input_)
