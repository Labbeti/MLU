
import torch
import unittest

from torch import Tensor
from typing import Sequence
from unittest import TestCase

from mlu.datasets.split.multilabel import (
	get_indexes_per_class_from_multihots,
	flat_indexes_per_class,
	get_multihots_from_indexes_per_class,
	balanced_split_v2,
)


def check_indexes_per_class(
	indexes_per_class: Sequence[Sequence[int]],
	at_least_one_elem_per_class: bool = True,
) -> bool:
	ok = True
	if at_least_one_elem_per_class:
		ok &= all(len(indexes) > 0 for indexes in indexes_per_class)
	return ok


def check_multihots(
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


class TestSplitMulti(TestCase):
	def test_convert(self):
		n_targets = 10000
		n_classes = 527

		targets = torch.rand(n_targets, n_classes).gt(0.5).bool()
		for target in targets:
			if target.sum().eq(False):
				target[torch.randint(n_classes, (1,))] = True

		indexes_per_class = get_indexes_per_class_from_multihots(targets)
		targets_rebuild = get_multihots_from_indexes_per_class(indexes_per_class)
		self.assertEqual(targets, targets_rebuild)

	def test_basic(self):
		targets = torch.as_tensor([
			[1, 1, 0],
			[1, 0, 0],
			[0, 0, 1],
			[1, 0, 1],
			[0, 1, 0],
			[0, 1, 1],
			[0, 1, 1],
			[1, 0, 0],
		])

		indexes_per_class = get_indexes_per_class_from_multihots(targets)
		expected_indexes_per_class = [
			[0, 1, 3, 7],
			[0, 4, 5, 6],
			[2, 3, 5, 6],
		]
		self.assertEqual(indexes_per_class, expected_indexes_per_class)

		indexes_all = list(range(len(targets)))
		indexes_per_class = get_indexes_per_class_from_multihots(targets)

		split_1, split_2 = balanced_split_v2(indexes_per_class, [0.5, 0.5], verbose=True)

		indexes_1 = flat_indexes_per_class(split_1)
		indexes_2 = flat_indexes_per_class(split_2)

		# print('Split indexes 1:', indexes_1)
		# print('Split indexes 2:', indexes_2)

		self.assertTrue(set(indexes_1).isdisjoint(set(indexes_2)))

		targets = targets.float()
		distribution = targets[indexes_all].mean(dim=0)
		distribution_1 = targets[indexes_1].mean(dim=0)
		distribution_2 = targets[indexes_2].mean(dim=0)

		# print('Distribution all:', distribution)
		# print('Distribution 1:', distribution_1)
		# print('Distribution 2:', distribution_2)

		def normalize(vec: Tensor) -> Tensor:
			return vec / vec.norm(p=1, keepdim=True)

		self.assertTrue(torch.allclose(normalize(distribution), normalize(distribution_1)))
		self.assertTrue(torch.allclose(normalize(distribution), normalize(distribution_2)))

	def test_random(self):
		n_targets = 1000
		n_classes = 10

		targets = torch.rand(n_targets, n_classes).gt(0.5)
		for target in targets:
			if target.sum().eq(0.0):
				target[torch.randint(n_classes, (1,))] = 1.0

		self.assertTrue(check_multihots(targets, at_least_one_elem_per_class=True, at_least_one_class_per_elem=True))

		indexes_per_class = get_indexes_per_class_from_multihots(targets)

		split_1, split_2 = balanced_split_v2(indexes_per_class, [0.5, 0.5], verbose=True)

		print('Split 1:', [len(indexes) for indexes in split_1])
		print('Split 1:', [len(indexes) for indexes in split_2])

		self.assertTrue(check_indexes_per_class(split_1, at_least_one_elem_per_class=True))
		self.assertTrue(check_indexes_per_class(split_2, at_least_one_elem_per_class=True))

		indexes_1 = flat_indexes_per_class(split_1)
		indexes_2 = flat_indexes_per_class(split_2)

		self.assertTrue(set(indexes_1).isdisjoint(set(indexes_2)))


if __name__ == '__main__':
	unittest.main()
