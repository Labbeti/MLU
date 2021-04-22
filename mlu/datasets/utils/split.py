
import numpy as np
import random
import torch

from numpy.random import RandomState
from torch import Tensor
from torch.utils.data.dataset import Dataset, Subset
from torch.utils.data.sampler import Sampler, SubsetRandomSampler
from types import ModuleType
from typing import List, Optional, Union

from mlu.utils.typing_ import SizedDataset


def generate_subsets_split(
	dataset: SizedDataset,
	n_classes: int,
	ratios: List[float],
	shuffle_idx: bool = True,
	target_one_hot: bool = False,
) -> List[Dataset]:
	"""
		Split mono-labeled dataset in several sub-wrappers by using a list of ratios.
		Also keep the original class distribution in every sub-dataset.

		:param dataset: The original dataset.
		:param n_classes: The number of classes in the original dataset.
		:param ratios: Ratios used to split the dataset. The sum must be 1.
		:param shuffle_idx: Shuffle classes indexes before split them.
		:param target_one_hot: Consider labels as one-hot vectors. If False, consider labels as class indexes.
		:return: A list of subsets.
	"""
	indexes = balanced_split(dataset, n_classes, ratios, shuffle_idx, target_one_hot)
	return [Subset(dataset, idx) for idx in indexes]


def generate_samplers_split(
	dataset: SizedDataset,
	n_classes: int,
	ratios: List[float],
	target_one_hot: bool = False,
) -> List[Sampler]:
	"""
		Split mono-labeled dataset with several samplers that must be used by pytorch Dataloaders.
		Also keep the original class distribution in every sub-dataset.

		:param dataset: The original dataset.
		:param n_classes: The number of classes in the original dataset.
		:param ratios: Ratios used to split the dataset. The sum must be 1.
		:param target_one_hot: Consider labels as one-hot vectors. If False, consider labels as class indexes.
		:return: A list of samplers of length of ratios list.
	"""
	indexes = balanced_split(dataset, n_classes, ratios, target_one_hot=target_one_hot)
	return [SubsetRandomSampler(idx) for idx in indexes]


def balanced_split(
	dataset: SizedDataset,
	n_classes: int,
	ratios: List[float],
	target_one_hot: bool = False,
	shuffle_idx: bool = True,
) -> List[List[int]]:
	"""
		Split dataset in list of indexes for each ratio.
		Also keep the original class distribution in every sub-dataset.

		:param dataset: The original dataset.
		:param n_classes: The number of classes in the original dataset.
		:param ratios: Ratios used to split the dataset. The sum must <= 1.
		:param target_one_hot: Consider labels as one-hot vectors. If False, consider labels as class indexes. (default: True)
		:param shuffle_idx: Shuffle classes indexes before split them. (default: True)
		:return: A list of indexes for each ratios.
	"""
	indexes_per_class = get_indexes_per_class(dataset, n_classes, target_one_hot)
	if shuffle_idx:
		indexes_per_class = shuffle_indexes_per_class(indexes_per_class)
	splits = split_indexes_per_class(indexes_per_class, ratios)
	indexes = [flat_indexes_per_class(split) for split in splits]
	return indexes


def get_indexes_per_class(
	dataset: SizedDataset,
	n_classes: int,
	target_one_hot: bool = False,
	label_index: int = 1,
) -> List[List[int]]:
	"""
		Get class indexes from a Sized dataset with index of class as label.

		:param dataset: The mono-labeled sized dataset to iterate.
		:param n_classes: The number of classes in the dataset.
		:param target_one_hot: If True, convert each label as one-hot label encoding instead of class index. (default: False)
		:param label_index: TODO
		:return: The indexes per class in the dataset of size (n_classes, n_elem_in_class_i).
			Note: If the class distribution is not perfectly uniform, this return is not a complete matrix.
	"""
	if not hasattr(dataset, '__len__'):
		raise RuntimeError('Dataset must have __len__() method for split indexes.')

	if hasattr(dataset, 'targets') and isinstance(dataset.targets, (np.ndarray, Tensor, list)):
		targets = dataset.targets
		targets = torch.as_tensor(targets)
		assert len(dataset) == len(targets), 'Dataset and targets must have the same len().'
	elif hasattr(dataset, 'get_target') and callable(dataset.get_target):
		targets = [torch.as_tensor(dataset.get_target(i)) for i in range(len(dataset))]
		targets = torch.stack(targets)
	else:
		targets = [torch.as_tensor(dataset[i][label_index]) for i in range(len(dataset))]
		targets = torch.stack(targets)

	if target_one_hot:
		targets = targets.argmax(dim=1)

	result = [
		torch.where(targets.eq(class_idx))[0].tolist()
		for class_idx in range(n_classes)
	]
	return result


def shuffle_indexes_per_class(
	indexes_per_class: List[List[int]],
	random_state: Optional[Union[RandomState, ModuleType]] = None,
) -> List[List[int]]:
	"""
		Shuffle each indexes per class. (this operation is 'in-place').

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
	flat_indexes: bool = False,
) -> Union[List[List[List[int]]], List[List[int]]]:
	"""
		Split distinct indexes per class.

		Example 1 :

		>>> indexes_per_class = get_indexes_per_class(dataset, n_classes=10)
		>>> indexes_s, indexes_u = split_indexes_per_class(indexes_per_class, [0.1, 0.9], flat_indexes=True)
		>>> subset_s = Subset(dataset, indexes_s)
		>>> subset_u = Subset(dataset, indexes_u)

		Example 2 :

		>>> split_indexes_per_class(indexes_per_class=[[1, 2], [3, 4], [5, 6]], ratios=[0.5, 0.5])
		... [[[1], [3], [5]], [[2], [4], [6]]]
		>>> dataset = Dataset()

		:param indexes_per_class: List of indexes of each class.
		:param ratios: The ratios of each indexes split.
		:param flat_indexes: If True, flat each sub-indexes. (default: False)
		:return: The indexes per ratio and per class of size (n_ratios, n_classes, n_indexes_in_ratio_and_class).
			Note: The return is not a tensor or ndarray because 'n_indexes_in_ratio_and_class' can be different for each
			ratio or class.
	"""
	assert 0.0 <= sum(ratios) <= 1.0, 'Ratio sum cannot be greater than 1.0.'

	n_classes = len(indexes_per_class)
	n_ratios = len(ratios)

	indexes_per_ratio_per_class = [[
			[] for _ in range(n_classes)
		]
		for _ in range(n_ratios)
	]

	current_starts = [0 for _ in range(n_classes)]
	for i, ratio in enumerate(ratios):
		for j, indexes in enumerate(indexes_per_class):
			current_start = current_starts[j]
			current_end = current_start + int(round(ratio * len(indexes)))
			sub_indexes = indexes[current_start:current_end]
			indexes_per_ratio_per_class[i][j] = sub_indexes
			current_starts[j] = current_end

	if flat_indexes:
		return flat_split_indexes_per_class(indexes_per_ratio_per_class)
	else:
		return indexes_per_ratio_per_class


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
