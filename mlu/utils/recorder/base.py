
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
			This method must be called at the end of an epoch.
		"""
		raise RuntimeError("Abstract method")

	def set_write_modes(self, write_iteration: bool, write_epoch: bool, write_global: bool):
		"""
			Activate or deactivate the writing modes in SummaryWriter.

			:param write_iteration: If True, write every iteration in writer.
			:param write_epoch: If True, write epochs values (defined by epochs trackers).
			:param write_global: If True, write global values (defined by global trackers).
		"""
		raise RuntimeError("Abstract method")

	def get_bests_epochs(self) -> Dict[str, Dict[str, Union[int, float]]]:
		"""
			:return: Best scores over all epochs on a dictionary like 'Dict[scalar_name][best_type] -> float'.
		"""
		raise RuntimeError("Abstract method")

	def get_writer(self) -> Optional[SummaryWriter]:
		"""
			:return: Internal SummaryWriter object.
		"""
		raise RuntimeError("Abstract method")
