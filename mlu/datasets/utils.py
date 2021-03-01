
import numpy as np
import random
import torch

from mlu.utils.typing import SizedDataset
from numpy.random import RandomState
from torch import Tensor
from torch.utils.data.dataset import Dataset, Subset
from torch.utils.data.sampler import Sampler, SubsetRandomSampler
from typing import Callable, List, Optional


def split_dataset(
	dataset: Dataset,
	nb_classes: int,
	ratios: List[float],
	shuffle_idx: bool = True,
	target_one_hot: bool = True,
) -> List[Dataset]:
	"""
		Split dataset in several sub-wrappers by using a list of ratios.
		Also keep the original class distribution in every sub-dataset.

		:param dataset: The original dataset.
		:param nb_classes: The number of classes in the original dataset.
		:param ratios: Ratios used to split the dataset. The sum must be 1.
		:param shuffle_idx: Shuffle classes indexes before split them.
		:param target_one_hot: Consider labels as one-hot vectors. If False, consider labels as class indexes.
		:returns: A list of sub-wrappers.
	"""
	indexes = generate_indexes(dataset, nb_classes, ratios, shuffle_idx, target_one_hot)
	return [Subset(dataset, idx) for idx in indexes]


def generate_split_samplers(
	dataset: Dataset,
	ratios: List[float],
	nb_classes: int,
	target_one_hot: bool = True,
) -> List[Sampler]:
	indexes = generate_indexes(dataset, nb_classes, ratios, target_one_hot=target_one_hot)
	return [SubsetRandomSampler(idx) for idx in indexes]


def generate_indexes(
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
		:returns: A list of indexes for each ratios.
	"""
	indexes_per_class = get_indexes_per_class(dataset, nb_classes, target_one_hot)
	if shuffle_idx:
		indexes_per_class = shuffle_indexes_per_class(indexes_per_class)
	indexes = split_indexes_per_class_flat(indexes_per_class, ratios)
	return indexes


def get_indexes_per_class(
	dataset: SizedDataset,
	nb_classes: int,
	target_one_hot: bool = True,
) -> List[List[int]]:
	"""
		Get class indexes from a Sized dataset with index of class as label.

		:param dataset: TODO
		:param nb_classes: TODO
		:param target_one_hot: TODO
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
	random_state: Optional[RandomState] = None,
) -> List[List[int]]:
	"""
		Shuffle each indexes per class. (this operation is "in-place").

		:param indexes_per_class: TODO
		:param random_state: TODO
		:return: TODO
	"""
	if random_state is None:
		random_state = random

	for indexes in indexes_per_class:
		random_state.shuffle(indexes)
	return indexes_per_class


def split_indexes_per_class_flat(
	indexes_per_class: List[List[int]],
	ratios: List[float],
) -> List[List[int]]:
	"""
		Split class indexes and merge them for each ratio.

		Ex:
		>>> split_indexes_per_class_flat(indexes_per_class=[[1, 2], [3, 4], [5, 6]], ratios=[0.5, 0.5])
		... [[1, 3, 5], [2, 4, 6]]

		:param indexes_per_class: TODO
		:param ratios: TODO
		:return: TODO
	"""
	assert sum(ratios) <= 1.0, "Ratio sum can be greater than 1.0."

	result = [[] for _ in range(len(ratios))]

	for indexes in indexes_per_class:
		current_begin = 0
		for j, ratio in enumerate(ratios):
			current_end = current_begin + int(round(ratio * len(indexes)))
			result[j] += indexes[current_begin:current_end]
			current_begin = current_end
	return result


def split_indexes_per_class(
	indexes_per_class: List[List[int]],
	ratios: List[float],
	round_fn: Callable[[float], int] = round,
) -> List[List[List[int]]]:
	"""
		Split class indexes.

		Ex:
		>>> split_indexes_per_class_flat(indexes_per_class=[[1, 2], [3, 4], [5, 6]], ratios=[0.5, 0.5])
		... [[[1], [3], [5]], [[2], [4], [6]]]

		:param indexes_per_class: List of indexes of each class.
		:param ratios: The ratios of each indexes split.
		:param round_fn: The round mode for compute the last index of a sub-indexes. (default: round)
		:return: The indexes per ratio and per class of size (nb_ratios, nb_classes, nb_indexes_in_ratio_and_class).
			Note: The return is not a tensor or ndarray because 'nb_indexes_in_ratio_and_class' can be different for each
			ratio or class.
	"""
	assert sum(ratios) <= 1.0, "Ratio sum can be greater than 1.0."

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


def reduce_indexes_per_class(
	indexes_per_class: List[List[int]],
	ratio: float,
) -> List[List[int]]:
	return split_indexes_per_class(indexes_per_class, [ratio])[0]


def flat_indexes_per_class(
	indexes_per_class: List[List[int]],
) -> List[int]:
	"""
		:param indexes_per_class: TODO
		:return: TODO
	"""
	indexes = []
	for class_indexes in indexes_per_class:
		indexes.extend(class_indexes)
	return indexes
