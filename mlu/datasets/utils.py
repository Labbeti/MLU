
import numpy as np
import random

from mlu.datasets.samplers import SubsetSampler
from torch.utils.data.dataset import Dataset, Subset
from torch.utils.data.sampler import Sampler
from typing import List


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


def generate_split_samplers(dataset: Dataset, ratios: List[float], nb_classes: int) -> List[Sampler]:
	indexes = generate_indexes(dataset, nb_classes, ratios, target_one_hot=True)
	return [SubsetSampler(dataset, idx) for idx in indexes]


def generate_indexes(
	dataset: Dataset,
	nb_classes: int,
	ratios: List[float],
	shuffle_idx: bool = True,
	target_one_hot: bool = True,
) -> List[List[int]]:
	"""
		Split dataset in list of indexes for each ratio.
		Also keep the original class distribution in every sub-dataset.

		:param dataset: The original dataset.
		:param nb_classes: The number of classes in the original dataset.
		:param ratios: Ratios used to split the dataset. The sum must be 1.
		:param shuffle_idx: Shuffle classes indexes before split them.
		:param target_one_hot: Consider labels as one-hot vectors. If False, consider labels as class indexes.
		:returns: A list of indexes for each ratios.
	"""
	cls_idx_all = _get_classes_idx(dataset, nb_classes, target_one_hot)
	if shuffle_idx:
		cls_idx_all = _shuffle_classes_idx(cls_idx_all)
	indexes = _split_classes_idx(cls_idx_all, ratios)
	return indexes


def _get_classes_idx(dataset: Dataset, nb_classes: int, target_one_hot: bool = True) -> List[List[int]]:
	"""
		Get class indexes from a standard dataset with index of class as label.
	"""
	result = [[] for _ in range(nb_classes)]

	for i in range(len(dataset)):
		_data, label = dataset[i]
		if target_one_hot:
			label = np.argmax(label)
		result[label].append(i)
	return result


def _shuffle_classes_idx(classes_idx: List[List[int]]) -> List[List[int]]:
	"""
		Shuffle each class indexes. (operation "in-place").
	"""
	for indexes in classes_idx:
		random.shuffle(indexes)
	return classes_idx


def _split_classes_idx(classes_idx: List[List[int]], ratios: List[float]) -> List[List[int]]:
	"""
		Split class indexes and merge them for each ratio.

		Ex:
			input:  classes_idx = [[1, 2], [3, 4], [5, 6]], ratios = [0.5, 0.5]
			output: [[1, 3, 5], [2, 4, 6]]
	"""

	result = [[] for _ in range(len(ratios))]

	for indexes in classes_idx:
		current_begin = 0
		for i, ratio in enumerate(ratios):
			current_end = current_begin + int(round(ratio * len(indexes)))
			result[i] += indexes[current_begin:current_end]
			current_begin = current_end
	return result
