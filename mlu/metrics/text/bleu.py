"""

The scratch code is based on the following sources :
	- https://github.com/Mjkim88/Pytorch-Torchtext-Seq2Seq/blob/master/bleu.py
	- https://github.com/neural-dialogue-metrics/BLEU/blob/master/bleu/metrics.py
	- https://github.com/tylin/coco-caption/blob/master/pycocoevalcap/bleu/bleu_scorer.py
	- https://github.com/amjha/BLEUCalculation/blob/master/calculatebleu3.py
"""

import torch

from collections import Counter
from torch import Tensor
from torchtext.data.metrics import bleu_score
from typing import List, Optional, Union

from mlu.metrics.base import Metric


class BLEU(Metric):
	"""
		BLEU metric score.

		Original paper : https://www.aclweb.org/anthology/P02-1040.pdf
	"""
	def __init__(
		self,
		ngram_order: int,
		smooth: bool = False,
		weights: Optional[List[float]] = None,
		backend: str = "scratch",
	):
		assert backend in ["scratch", "torchtext"], f"Supported backends are {('scratch', 'torchtext')}."

		if backend == "scratch":
			if weights is not None:
				print(f"WARNING: Weights argument is not supported for '{backend}' backend, it will be ignored.")
		elif backend == "torchtext":
			if smooth:
				print(f"WARNING: smooth=True is not supported for '{backend}' backend, it will be ignored.")

		super().__init__()
		self.ngram_order = ngram_order
		self.smooth = smooth
		self.weights = weights if weights is not None else (torch.ones(self.ngram_order) / self.ngram_order).tolist()
		self.backend = backend

	def compute_score(
		self,
		candidate_corpus: Union[List[Tensor], List[List[str]]],
		references_corpus: Union[List[List[Tensor]], List[List[List[str]]]],
	) -> Tensor:
		"""
			Compute the BLEU score metric.

			:param candidate_corpus: (N, candidate_corpus)
			:param references_corpus: (N, nb references, reference size)
			:return: The BLEU score in range [0.0, 1.0].
		"""
		if self.backend == "scratch":
			return compute_bleu_score_scratch(candidate_corpus, references_corpus, self.ngram_order, self.smooth)
		elif self.backend == "torchtext":
			return bleu_score(candidate_corpus, references_corpus, self.ngram_order, self.weights)
		else:
			raise RuntimeError(f"Unknown backend '{self.backend}' for BLEU metric.")


def compute_bleu_score_scratch(
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

	# Compute geometric mean
	if min(precisions) > 0.0:
		weights = torch.full_like(precisions, fill_value=1.0 / max_order)
		geo_mean = (precisions.log() * weights).sum().exp()
	else:
		geo_mean = torch.zeros(1)

	# Compute brevity penalty
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


def get_ngrams(sentence: Union[Tensor, List[str]], max_order: int, ignored_value: Optional[float] = None) -> Counter:
	counter = Counter()
	for order in range(1, max_order + 1):
		for i in range(len(sentence) - order + 1):
			if isinstance(sentence, Tensor):
				ngram = sentence[i:i + order].tolist()
			else:
				ngram = sentence[i:i + order]
			if ignored_value is None or ignored_value not in ngram:
				counter[tuple(ngram)] += 1
	return counter
