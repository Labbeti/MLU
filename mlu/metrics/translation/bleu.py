
import torch

from collections import Counter
from mlu.metrics.base import Metric
from torch import Tensor
from typing import List, Optional


class BLEU(Metric):
	"""
		BLEU metric score.

		Original paper : https://www.aclweb.org/anthology/P02-1040.pdf
	"""
	def __init__(self, ngram_order: int, smooth: bool = False):
		super().__init__()
		self.ngram_order = ngram_order
		self.smooth = smooth

	def compute_score(self, candidate_corpus: List[Tensor], references_corpus: List[List[Tensor]]) -> Tensor:
		"""
			Compute the BLEU score metric.

			The code is based on the following codes :
				- https://github.com/Mjkim88/Pytorch-Torchtext-Seq2Seq/blob/master/bleu.py
				- https://github.com/neural-dialogue-metrics/BLEU/blob/master/bleu/metrics.py
				- https://github.com/tylin/coco-caption/blob/master/pycocoevalcap/bleu/bleu_scorer.py
				- https://github.com/amjha/BLEUCalculation/blob/master/calculatebleu3.py

			:param candidate_corpus: (N, candidate_corpus)
			:param references_corpus: (N, nb references, reference size)
			:return: The BLEU score in range [0.0, 1.0].
		"""
		return compute_bleu_score(candidate_corpus, references_corpus, self.ngram_order, self.smooth)


def compute_bleu_score(
	candidate_corpus: List[Tensor],
	references_corpus: List[List[Tensor]],
	max_order: int = 4,
	smooth: bool = False,
) -> Tensor:
	assert len(candidate_corpus) == len(references_corpus)

	if len(candidate_corpus) == 0:
		return torch.zeros(1)

	occ_by_order = torch.zeros(max_order)
	occ_by_order_max = torch.zeros(max_order)

	for candidate, references in zip(candidate_corpus, references_corpus):
		counter_input = get_ngrams(candidate, max_order)
		counter_target = get_ngrams_max_counter(references, max_order)

		# The operator "&" merge dicts (intersection) and apply min between the two value stored
		counter_min = counter_input & counter_target

		for ngram, occurrence in counter_min.items():
			order = len(ngram) - 1
			occ_by_order[order] += occurrence

		occ_by_order_max += torch.as_tensor([len(candidate) - n for n in range(max_order)])

	# Compute precisions
	if not smooth:
		precisions = occ_by_order / occ_by_order_max
	else:
		precisions = (occ_by_order + 1.0) / (occ_by_order_max + 1.0)

	if min(precisions) > 0.0:
		weights = torch.full_like(precisions, fill_value=1.0 / max_order)
		geo_mean = (precisions.log() * weights).sum().exp()
	else:
		geo_mean = torch.zeros(1)

	candidates_len = sum(len(candidate) for candidate in candidate_corpus)
	references_len = sum(min([len(reference) for reference in references]) for references in references_corpus)

	if candidates_len < references_len:
		bp = 1.0 - references_len / candidates_len
		bp = torch.scalar_tensor(bp)
		bp = bp.exp()
	else:
		bp = 1.0

	score = geo_mean * bp
	return score


def get_ngrams_max_counter(sentences: List[Tensor], max_order: int, ignored_value: Optional[float] = None) -> Counter:
	counter_max = Counter()
	for sentence in sentences:
		counter = get_ngrams(sentence, max_order, ignored_value)
		# The operator "|" merge dicts (union) and apply max if each counters has the same key
		counter_max |= counter
	return counter_max


def get_ngrams(sentence: Tensor, max_order: int, ignored_value: Optional[float] = None) -> Counter:
	counter = Counter()
	for order in range(1, max_order + 1):
		for i in range(len(sentence) - order + 1):
			ngram = sentence[i:i + order].tolist()
			if ignored_value is None or ignored_value not in ngram:
				counter[tuple(ngram)] += 1
	return counter
