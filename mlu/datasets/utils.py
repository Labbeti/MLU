
import numpy as np
import random
import torch

from mlu.utils.typing import SizedDataset
from numpy.random import RandomState
from torch import Tensor
from torch.utils.data.dataset import Dataset, Subset
from torch.utils.data.sampler import Sampler, SubsetRandomSampler
from types import ModuleType
from typing import Callable, List, Optional, Union


def generate_subsets_split(
	dataset: Dataset,
	nb_classes: int,
	ratios: List[float],
	shuffle_idx: bool = True,
	target_one_hot: bool = False,
) -> List[Dataset]:
	"""
		Split mono-labeled dataset in several sub-wrappers by using a list of ratios.
		Also keep the original class distribution in every sub-dataset.

		:param dataset: The original dataset.
		:param nb_classes: The number of classes in the original dataset.
		:param ratios: Ratios used to split the dataset. The sum must be 1.
		:param shuffle_idx: Shuffle classes indexes before split them.
		:param target_one_hot: Consider labels as one-hot vectors. If False, consider labels as class indexes.
		:return: A list of subsets.
	"""
	indexes = generate_indexes_split(dataset, nb_classes, ratios, shuffle_idx, target_one_hot)
	return [Subset(dataset, idx) for idx in indexes]


def generate_samplers_split(
	dataset: Dataset,
	nb_classes: int,
	ratios: List[float],
	target_one_hot: bool = False,
) -> List[Sampler]:
	"""
		Split mono-labeled dataset with several samplers that must be used by pytorch Dataloaders.
		Also keep the original class distribution in every sub-dataset.

		:param dataset: The original dataset.
		:param nb_classes: The number of classes in the original dataset.
		:param ratios: Ratios used to split the dataset. The sum must be 1.
		:param target_one_hot: Consider labels as one-hot vectors. If False, consider labels as class indexes.
		:return: A list of samplers of length of ratios list.
	"""
	indexes = generate_indexes_split(dataset, nb_classes, ratios, target_one_hot=target_one_hot)
	return [SubsetRandomSampler(idx) for idx in indexes]


def generate_indexes_split(
	dataset: Dataset,
	nb_classes: int,
	ratios: List[float],
	target_one_hot: bool = True,
	shuffle_idx: bool = True,
) -> List[List[int]]:
	"""
		Split dataset in list of indexes for each ratio.
		Also keep the original class distribution in every sub-dataset.

		:param dataset: The original dataset.
		:param nb_classes: The number of classes in the original dataset.
		:param ratios: Ratios used to split the dataset. The sum must <= 1.
		:param target_one_hot: Consider labels as one-hot vectors. If False, consider labels as class indexes. (default: True)
		:param shuffle_idx: Shuffle classes indexes before split them. (default: True)
		:return: A list of indexes for each ratios.
	"""
	indexes_per_class = get_indexes_per_class(dataset, nb_classes, target_one_hot)
	if shuffle_idx:
		indexes_per_class = shuffle_indexes_per_class(indexes_per_class)
	splits = split_indexes_per_class(indexes_per_class, ratios)
	indexes = [flat_indexes_per_class(split) for split in splits]
	return indexes


def get_indexes_per_class(
	dataset: SizedDataset,
	nb_classes: int,
	target_one_hot: bool = False,
) -> List[List[int]]:
	"""
		Get class indexes from a Sized dataset with index of class as label.

		:param dataset: The mono-labeled sized dataset to iterate.
		:param nb_classes: The number of classes in the dataset.
		:param target_one_hot: If True, convert each label as one-hot label encoding instead of class index. (default: False)
		:return: The indexes per class in the dataset of size (num_classes, num_elem_in_class_i).
			Note: If the class distribution is not perfectly uniform, this return is not a complete matrix.
	"""
	result = [[] for _ in range(nb_classes)]

	for i in range(len(dataset)):
		_data, label = dataset[i]
		if target_one_hot:
			if isinstance(label, np.ndarray) or isinstance(label, Tensor):
				label_idx = label.argmax().item()
			else:
				raise RuntimeError(
					f"Invalid one-hot label type '{type(label)}' at index {i}. Must be one of '{(np.ndarray, torch.Tensor)}'")
		else:
			label_idx = label
		result[label_idx].append(i)
	return result


def shuffle_indexes_per_class(
	indexes_per_class: List[List[int]],
	random_state: Optional[Union[RandomState, ModuleType]] = None,
) -> List[List[int]]:
	"""
		Shuffle each indexes per class. (this operation is "in-place").

		:param indexes_per_class: The list of indexes per class.
		:param random_state: The module or numpy RandomState to use for shuffle. If None, use python random module.
		:return: The list of indexes per class shuffled.
	"""
	if random_state is None:
		random_state = random

	for indexes in indexes_per_class:
		random_state.shuffle(indexes)
	return indexes_per_class


def split_indexes_per_class(
	indexes_per_class: List[List[int]],
	ratios: List[float],
	round_fn: Callable[[float], int] = round,
) -> List[List[List[int]]]:
	"""
		Split distinct indexes per class.

		Ex:

		>>> split_indexes_per_class(indexes_per_class=[[1, 2], [3, 4], [5, 6]], ratios=[0.5, 0.5])
		... [[[1], [3], [5]], [[2], [4], [6]]]

		:param indexes_per_class: List of indexes of each class.
		:param ratios: The ratios of each indexes split.
		:param round_fn: The round mode for compute the last index of a sub-indexes. (default: round)
		:return: The indexes per ratio and per class of size (nb_ratios, nb_classes, nb_indexes_in_ratio_and_class).
			Note: The return is not a tensor or ndarray because 'nb_indexes_in_ratio_and_class' can be different for each
			ratio or class.
	"""
	assert 0.0 <= sum(ratios) <= 1.0, "Ratio sum cannot be greater than 1.0."

	nb_classes = len(indexes_per_class)
	nb_ratios = len(ratios)

	indexes_per_ratio_per_class = [[
			[] for _ in range(nb_classes)
		]
		for _ in range(nb_ratios)
	]

	current_starts = [0 for _ in range(nb_classes)]
	for i, ratio in enumerate(ratios):
		for j, indexes in enumerate(indexes_per_class):
			current_start = current_starts[j]
			current_end = current_start + int(round_fn(ratio * len(indexes)))
			sub_indexes = indexes[current_start:current_end]
			indexes_per_ratio_per_class[i][j] = sub_indexes
			current_starts[j] = current_end
	return indexes_per_ratio_per_class


def split_multilabel_indexes_per_class(
	indexes_per_class: List[List[int]],
	ratios: List[float],
) -> List[List[List[int]]]:
	raise NotImplementedError("TODO")


def reduce_indexes_per_class(
	indexes_per_class: List[List[int]],
	ratio: float,
) -> List[List[int]]:
	return split_indexes_per_class(indexes_per_class, [ratio])[0]


def flat_split_indexes_per_class(splits: List[List[List[int]]]) -> List[List[int]]:
	return [flat_indexes_per_class(split) for split in splits]


def flat_indexes_per_class(
	indexes_per_class: List[List[int]],
) -> List[int]:
	"""
		:param indexes_per_class: The indexes per class.
		:return: The complete indexes list of the indexes per class.
	"""
	indexes = []
	for class_indexes in indexes_per_class:
		indexes.extend(class_indexes)
	return indexes
