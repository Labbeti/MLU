
from typing import Dict, Iterable, Optional, Set, Union


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
