
import numpy as np
import torch

from torch import Tensor
from torch.nn.functional import one_hot
from typing import List, Union


# --- ONE-HOT ---
def nums_to_onehot(nums: Union[np.ndarray, Tensor], nb_classes: int) -> Union[np.ndarray, Tensor]:
	"""
		Convert numbers (or indexes) of classes to one-hot version.

		:param nums: Label of indexes of classes.
		:param nb_classes: The maximum number of distinct classes.
		:returns: Label with one-hot vectors
	"""
	if isinstance(nums, Tensor):
		onehot_vectors = one_hot(nums, nb_classes)
	elif isinstance(nums, np.ndarray):
		onehot_vectors = nums_to_onehot(torch.from_numpy(nums), nb_classes).cpu().numpy()
	else:
		onehot_vectors = nums_to_onehot(torch.as_tensor(nums), nb_classes)
	return onehot_vectors


def nums_to_smooth_onehot(nums: Union[np.ndarray, Tensor], nb_classes: int, smooth: float) -> Union[np.ndarray, Tensor]:
	"""
		Convert numbers (or indexes) of classes to smooth one-hot version.

		:param nums: Label of indexes of classes.
		:param nb_classes: The maximum number of distinct classes.
		:param smooth: The label smoothing coefficient in [0, 1/nb_classes].
		:returns: Label with smooth one-hot vectors
	"""
	onehot_vectors = nums_to_onehot(nums, nb_classes)
	return onehot_to_smooth_onehot(onehot_vectors, nb_classes, smooth)


def onehot_to_nums(onehot_vectors: Tensor) -> Tensor:
	""" Convert a list of one-hot vectors of size (N, C) to a list of classes numbers of size (N). """
	return onehot_vectors.argmax(dim=1)


def onehot_to_smooth_onehot(
	onehot_vectors: Union[np.ndarray, Tensor], nb_classes: int, smooth: float
) -> Union[np.ndarray, Tensor]:
	""" Smooth one-hot labels with a smoothing coefficient. """
	classes_smoothed = (1.0 - smooth) * onehot_vectors + smooth / nb_classes
	return classes_smoothed


def binarize_pred_to_onehot(pred: Tensor, dim: int = 1) -> Tensor:
	""" Convert a batch of labels (bsize, label_size) to one-hot by using max(). """
	indexes = pred.argmax(dim=dim)
	nb_classes = pred.shape[dim]
	onehot_vectors = one_hot(indexes, nb_classes)
	return onehot_vectors


# --- MULTI-HOT ---
def nums_to_matrix(nums_lst: List[List[int]], fill_value: float = -1.0) -> Tensor:
	bsize = len(nums_lst)
	nb_classes = 0
	for nums in nums_lst:
		nb_classes = max(nb_classes, len(nums))
	result = torch.full(size=(bsize, nb_classes), fill_value=fill_value)
	for i, nums in enumerate(nums_lst):
		for j, num in enumerate(nums):
			result[i, j] = num
	return result


def nums_to_multihot(nums_lst: List[List[Union[int, float]]], nb_classes: int) -> Tensor:
	"""
		Convert a list of numbers (or indexes) of classes to multi-hot version.

		:param nums_lst: List of List of indexes of classes.
		:param nb_classes: The maximum number of classes.
		:returns: Label with multi-hot vectors
	"""
	result = torch.empty(len(nums_lst), nb_classes)
	for i, nums in enumerate(nums_lst):
		label = torch.zeros(nb_classes)
		for num in nums:
			label[num] = 1.0
		result[i] = label
	return result


def multihot_to_nums(multihots: Union[np.ndarray, Tensor], threshold: float = 1.0) -> List[List[int]]:
	"""
		Convert multi-hot vectors to a list of list of classes indexes.

		:param multihots: The multi-hot vectors of shape (bsize, nb classes).
		:param threshold: The threshold used to determine if class is present or not.
		:returns: The list of list of classes indexes. Each sub-list can have a different size.
	"""
	result = [
		[j for j, coefficient in enumerate(label) if coefficient >= threshold]
		for i, label in enumerate(multihots)
	]
	return result


def multihot_to_smooth_multihot(
	multihot_vectors: Union[np.ndarray, Tensor],
	smooth: float,
) -> Union[np.ndarray, Tensor]:
	"""
		Smooth multi-hot labels with a smoothing coefficient.

		:param multihot_vectors: Multi-hot vectors of shape (bsize, nb classes).
		:param nb_classes: The maximum number of classes.
		:param smooth: The label smoothing coefficient in [0, 1].
		:returns: The smoothed multi-hot vectors.
	"""
	nb_classes = multihot_vectors.shape[-1]
	result = (1.0 - smooth) * multihot_vectors + smooth / nb_classes
	return result
