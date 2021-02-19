
from abc import ABC
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, Optional, Union


class RecorderABC(ABC):
	def add_scalar(self, scalar_name: str, scalar: Union[float, Tensor], iteration: int, epoch: int):
		"""
			Add scalar value to recorder.

			:param scalar_name: The name of the scalar value.
			:param scalar: The value to store.
			:param iteration: The current iteration index.
			:param epoch: The current epoch.
		"""
		raise RuntimeError("Abstract method")

	def step(self):
		"""
			Update global trackers and save epoch results to tensorboard.
		"""
		raise RuntimeError("Abstract method")

	def get_writer(self) -> Optional[SummaryWriter]:
		"""
			:return: Internal SummaryWriter object.
		"""
		raise RuntimeError("Abstract method")

	def get_all_bests(self) -> Dict[str, Dict[str, float]]:
		"""
			:return: Best score over epochs on a dictionary Dict[scalar_name][best_type].
		"""
		raise RuntimeError("Abstract method")
