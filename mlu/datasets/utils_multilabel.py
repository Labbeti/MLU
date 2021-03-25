
import torch

from torch import Tensor
from typing import List


def get_indexes_per_class(targets: Tensor) -> List[List[int]]:
	num_classes = targets.shape[1]
	return [torch.where(targets[:, i].eq(1.0))[0].tolist() for i in range(num_classes)]


def get_targets(indexes_per_class: List[List[int]]) -> Tensor:
	max_idx = -1
	for indexes in indexes_per_class:
		max_idx = max(max_idx, max(indexes))

	num_classes = len(indexes_per_class)
	targets = torch.full((max_idx + 1, num_classes), fill_value=False, dtype=torch.bool)
	for idx_class, indexes in enumerate(indexes_per_class):
		for idx in indexes:
			targets[idx, idx_class] = True
	return targets


def flat_indexes_per_class(indexes_per_class: List[List[int]]) -> List[int]:
	all_indexes = []
	for indexes in indexes_per_class:
		all_indexes += indexes
	return torch.unique(torch.as_tensor(all_indexes, dtype=torch.long)).tolist()


def split_multilabel_indexes_per_class(
	indexes_per_class: List[List[int]],
	ratios: List[float],
	verbose: bool = False,
) -> List[List[List[int]]]:
	targets = get_targets(indexes_per_class)
	indexes_per_class = [torch.as_tensor(indexes) for indexes in indexes_per_class]

	num_classes = len(indexes_per_class)
	num_splits = len(ratios)
	num_elements = targets.shape[0]

	if verbose:
		print(f"Info: num_classes={num_classes}, num_splits={num_splits}, num_elements={num_elements}")

	num_expected_per_splits = torch.as_tensor([
		[round(len(indexes) * ratio) for indexes in indexes_per_class]
		for ratio in ratios
	])

	splits = [[
			[] for _ in range(num_classes)
		]
		for _ in range(num_splits)
	]

	taken = torch.full((num_elements,), False, dtype=torch.bool)
	num_taken = 0

	while num_taken < num_elements:
		num_by_splits = torch.as_tensor([[len(indexes) for indexes in split] for split in splits])
		num_missing_per_splits = num_expected_per_splits - num_by_splits

		if verbose:
			num_missing_total = num_missing_per_splits.sum().item()
			print(f"[{num_taken}/{num_elements}] taken. Missing: {num_missing_total}. ", end="\r")

		# Search the max missing elem
		idx_ratio_prior, idx_class_prior = torch.where(num_missing_per_splits.eq(num_missing_per_splits.max()))
		random_prior = torch.randint(len(idx_ratio_prior), (1, ))
		idx_ratio_prior = idx_ratio_prior[random_prior].item()
		idx_class_prior = idx_class_prior[random_prior].item()

		# Search if one elem of this class is available
		indexes = indexes_per_class[idx_class_prior]
		taken_class = torch.where(taken[indexes].eq(False))[0]

		if len(taken_class) > 0:
			random_in_class = torch.randint(len(taken_class), (1, ))
			found_idx = indexes[taken_class[random_in_class]]

			# Search the other classes of this elem "found_idx"
			classes_of_found_idx = torch.where(targets[found_idx])[0].tolist()

			for idx_class in classes_of_found_idx:
				splits[idx_ratio_prior][idx_class].append(found_idx)
			taken[found_idx] = True
			num_taken += 1

		else:
			# If no elem of the class search is available, ignore this class now
			num_expected_per_splits[idx_ratio_prior][idx_class_prior] = num_by_splits[idx_ratio_prior][idx_class_prior]

	if verbose:
		print(f"[{num_taken}/{num_elements}] taken.", end="\n")
	return splits


def check_indexes_per_class(
	indexes_per_class: List[List[int]],
	at_least_one_elem_per_class: bool = True,
) -> bool:
	ok = True
	if at_least_one_elem_per_class:
		ok &= all(len(indexes) > 0 for indexes in indexes_per_class)
	return ok


def check_targets(
	targets: Tensor,
	at_least_one_elem_per_class: bool = True,
	at_least_one_class_per_elem: bool = True,
) -> bool:
	ok = True
	if at_least_one_elem_per_class:
		ok &= targets.sum(dim=0).gt(0).all().item()
	if at_least_one_class_per_elem:
		ok &= targets.sum(dim=1).gt(0).all().item()
	return ok
