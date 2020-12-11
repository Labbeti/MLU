
import torch

from torch import Tensor
from typing import Dict, Iterable, List, Optional, Set, Union


def sentence_to_tensor(
	sentence: List[str],
	word_to_cls_table: Optional[Dict[str, int]] = None,
) -> Tensor:
	index_encoding = torch.zeros(len(sentence))
	if word_to_cls_table is None:
		word_to_cls_table = {}

	for i, word in enumerate(sentence):
		if word not in word_to_cls_table.keys():
			word_to_cls_table[word] = len(word_to_cls_table)
		index_encoding[i] = word_to_cls_table[word]

	return index_encoding


def list_sentence_to_list_tensor(
	sentences: List[List[str]],
	word_to_cls_table: Dict[str, int],
) -> List[Tensor]:
	return [sentence_to_tensor(sentence, word_to_cls_table) for sentence in sentences]


def tensor_to_sentence(tensor: Tensor, cls_to_word_table: Dict[int, str]) -> List[str]:
	sentence = ["" for _ in range(len(tensor))]
	for i, idx in enumerate(tensor):
		sentence[i] = cls_to_word_table[idx.item()]
	return sentence


def add_to_vocabulary(text: Union[str, Iterable], vocabulary: Optional[Set[str]] = None) -> Set[str]:
	if vocabulary is None:
		vocabulary = set()

	if isinstance(text, str):
		vocabulary.add(text)
	else:
		for subtext in text:
			add_to_vocabulary(subtext, vocabulary)

	return vocabulary


def build_conversions_tables(vocabulary: Set[str]) -> (Dict[str, int], Dict[int, str]):
	word_to_cls_table, cls_to_word_table = {}, {}
	for i, word in enumerate(vocabulary):
		word_to_cls_table[word] = i
		cls_to_word_table[i] = word
	return word_to_cls_table, cls_to_word_table


def list_tensor_to_tensor(lst: List[Tensor], fill_value: float) -> Tensor:
	max_length = max([len(subtensor) for subtensor in lst])
	results = [torch.cat((subtensor, torch.full([max_length - len(subtensor)], fill_value))) for subtensor in lst]
	results = torch.stack(results)
	return results
