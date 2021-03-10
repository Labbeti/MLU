
from torch.utils.data.dataset import Dataset
from typing import Iterable, Protocol, Sized, Union


class SizedIterable(Sized, Iterable, Protocol):
	"""
		Class that inherit from Sized and Iterable protocols classes.

		Subclasses must implements '__iter__' and '__len__' methods.
	"""
	pass


class SizedDatasetProtocol(Sized, Protocol):
	"""
		Class that inherit from Sized and add the '__getitem__' method of a dataset.

		Subclasses must implements '__getitem__' and '__len__' methods.
	"""
	def __getitem__(self, idx):
		raise NotImplementedError("Abstract method")

	def __len__(self) -> int:
		raise NotImplementedError("Abstract method")


SizedDataset = Union[Dataset, SizedDatasetProtocol]
