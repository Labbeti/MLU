
import torch

from torch import Tensor

from mlu.metrics.base import Metric
from mlu.metrics.metric import Precision, Recall
from nltk.translate.meteor_score import meteor_score
from torch.nn.functional import one_hot
from typing import Optional, List


class METEOR(Metric[str, List[str], Tensor]):
	def __init__(self, alpha: float = 0.9, gamma: float = 0.5, beta: float = 3.0):
		super().__init__()
		self.alpha = alpha
		self.gamma = gamma
		self.beta = beta

	def compute_score(self, candidate: str, references: List[str]) -> Tensor:
		score = meteor_score(references, candidate, alpha=self.alpha, beta=self.beta, gamma=self.gamma)
		return torch.scalar_tensor(score)


class METEORV2(Metric):
	""" TODO : test """

	def __init__(self, nb_classes: Optional[int] = None, alpha: float = 0.9, gamma: float = 0.5, beta: float = 3.0):
		super().__init__()
		self.nb_classes = nb_classes
		self.alpha = alpha
		self.gamma = gamma
		self.beta = beta

		self.recall = Recall()
		self.precision = Precision()

	def compute_score(self, candidate: Tensor, references: List[Tensor]) -> Tensor:
		if self.nb_classes is None:
			idx_set = set(candidate.unique().tolist())
			for reference in references:
				idx_set.union(reference.unique().tolist())
			nb_classes = len(idx_set)
		else:
			nb_classes = self.nb_classes

		print("candidate ", candidate)
		print("references", references)
		print("nb_classes", nb_classes)

		candidate_onehot = one_hot(candidate.to(torch.int64), nb_classes)
		references_onehot = [one_hot(reference.to(torch.int64), nb_classes) for reference in references]

		max_sentence_size = max([len(ref) for ref in references_onehot] + [len(candidate_onehot)])
		print("max_sentence_size", max_sentence_size)
		print("shapes", candidate_onehot.shape, " ; ", [ref.shape for ref in references_onehot])
		candidate_onehot = torch.cat((candidate_onehot, torch.zeros(max_sentence_size - candidate_onehot.shape[0], nb_classes)))
		references_onehot = [torch.cat((reference, torch.zeros(max_sentence_size - reference.shape[0], nb_classes))) for reference in references_onehot]

		recall = [self.recall(candidate_onehot, reference) for reference in references_onehot]
		precision = [self.precision(candidate_onehot, reference) for reference in references_onehot]

		print("recall", recall)
		print("precision", precision)

		recall = torch.as_tensor(recall)
		precision = torch.as_tensor(precision)

		numerator = precision * recall
		denominator = self.alpha * precision + (1.0 - self.alpha) * recall

		scores = torch.as_tensor([n / d if d != 0.0 else 0.0 for n, d in zip(numerator, denominator)])

		nb_chunks = get_nb_chunks(candidate, references)
		nb_matches = get_nb_matches(candidate, references)
		print("nb_chunks", nb_chunks)
		print("nb_matches", nb_matches)

		frag = nb_chunks / nb_matches
		pen = self.gamma * frag ** self.beta
		scores = (1.0 - pen) * scores

		score = scores.max()
		print("score", score)

		return score


def get_nb_matches(candidate: Tensor, references: List[Tensor]) -> Tensor:
	nb_matches = [sum(1 if word in ref else 0 for word in candidate) for ref in references]
	return torch.as_tensor(nb_matches)


def get_nb_chunks(candidate: Tensor, references: List[Tensor]) -> Tensor:
	nb_chunks = torch.as_tensor([
		divide_in_chunks(candidate, reference)
		for reference in references
	])
	return nb_chunks


def divide_in_chunks(candidate: Tensor, reference: Tensor) -> int:
	i = 0
	nb_chunks = 0
	while i < len(candidate):
		sub_seq_len = 1
		continue_ = True
		prev_start = -1
		while continue_ and i + sub_seq_len <= len(candidate):
			sub_seq = candidate[i:i + sub_seq_len]
			if prev_start == -1:
				continue_, start = contains_sub_seq(reference, sub_seq)
			else:
				if prev_start + len(sub_seq) > len(reference):
					continue_ = False
				else:
					continue_ = reference[prev_start:prev_start+len(sub_seq)].eq(sub_seq).all()
			print(f"{i} : {reference.tolist()}, {candidate.tolist()}, {sub_seq.tolist()}, {continue_}, {start}")
			if not continue_ and prev_start != -1:
				indexes = torch.as_tensor(list(range(prev_start)) + list(range(prev_start+len(sub_seq)-1, len(reference)))).long()
				print(f"indexes {indexes.tolist()}")
				reference = reference[indexes]
			prev_start = start
			sub_seq_len += 1
		i += sub_seq_len - 1
		nb_chunks += 1

	return nb_chunks


def contains_sub_seq(seq: Tensor, sub_seq: Tensor) -> (bool, int):
	found = False
	start = -1
	for i in range(len(seq)):
		found = True
		for j in range(len(sub_seq)):
			if i+j >= len(seq) or seq[i+j] != sub_seq[j]:
				found = False
				break
		if found:
			start = i
			break
	return found, start
