
from abc import ABC
from typing import Iterable, Protocol, Sized, runtime_checkable
from torch.utils.data.dataset import Dataset


class SizedIterable(Sized, Iterable, Protocol):
	"""
		Class that inherit from Sized and Iterable protocols classes.

		Subclasses must implements '__iter__' and '__len__' methods.
	"""
	pass


@runtime_checkable
class SizedDatasetLike(Protocol):
	"""
		Class that inherit from Sized and add the '__getitem__' method of a dataset.

		Subclasses must implements '__getitem__' and '__len__' methods.
	"""
	def __getitem__(self, idx):
		raise NotImplemented("Abstract method")

	def __len__(self) -> int:
		raise NotImplemented("Abstract method")


class SizedDataset(Dataset, ABC):
	def __len__(self) -> int:
		raise NotImplemented("Abstract method")

	@staticmethod
	def __instancecheck__(obj) -> bool:
		return SizedDataset.__subclasscheck__(type(obj))

	@staticmethod
	def __subclasscheck__(cls) -> bool:
		if issubclass(cls, Dataset) and hasattr(cls, "__len__"):
			return True
		else:
			return False
