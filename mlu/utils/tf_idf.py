
import torch

from torch import Tensor
from typing import Callable, List


class TF(Callable):
	def __call__(self, word: Tensor, document: Tensor) -> Tensor:
		return term_frequency(word, document)


class IDF(Callable):
	def __call__(self, word: Tensor, documents: List[Tensor]) -> Tensor:
		return inverse_document_frequency(word, documents)


class TFIDF(Callable):
	def __init__(self):
		super().__init__()
		self.tf = TF()
		self.idf = IDF()

	def __call__(self, word: Tensor, documents: List[Tensor]) -> Tensor:
		idf_score = self.idf(word, documents)
		tf_scores = torch.as_tensor([self.tf(word, doc) for doc in documents])
		return tf_scores * idf_score


def inverse_document_frequency(word: Tensor, documents: List[Tensor]) -> Tensor:
	n_docs = len(documents)
	n_docs_with_word = sum([1 for doc in documents if word in doc])
	score = torch.scalar_tensor(n_docs / n_docs_with_word).log()
	return score


def term_frequency(word: Tensor, document: Tensor) -> Tensor:
	count = document.eq(word).sum()
	n_docs = len(document)
	return count / n_docs
