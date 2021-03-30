
from abc import ABC
from torch.utils.data.dataloader import DataLoader


class TrainerABC(ABC):
	def __init__(self, epoch_start: int = 0, it_start: int = 0):
		super().__init__()
		self._it_start = it_start
		self._epoch_start = epoch_start

		self._it_end = None
		self._epoch_end = None

	def train(self, nb_epochs: int):
		self._it_end = self._it_start
		self._epoch_end = self._epoch_start

		for epoch in range(self._epoch_start, self._epoch_start + nb_epochs):
			self.train_epoch(epoch, nb_epochs)
			self._epoch_end += 1

	def train_epoch(self, epoch: int, nb_epochs: int):
		for it, item in enumerate(self.get_loader()):
			self.train_it(item, it, len(self.get_loader()), epoch, nb_epochs)
			self._it_end += 1

	def train_it(self, item: tuple, it: int, nb_it: int, epoch: int, nb_epochs: int):
		raise NotImplemented("Abstract method")

	def get_loader(self) -> DataLoader:
		raise NotImplemented("Abstract method")
