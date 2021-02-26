
from mlu.utils.recorder.base import RecorderABC
from torch import Tensor
from torch.utils.tensorboard.writer import SummaryWriter
from typing import Dict, Optional, Union


class EmptyRecorder(RecorderABC):
	def add_scalar(self, scalar_name: str, scalar: Union[float, Tensor], iteration: int, epoch: int):
		pass

	def step(self):
		pass

	def get_bests_epochs(self) -> Dict[str, Dict[str, Union[int, float]]]:
		return {}

	def get_writer(self) -> Optional[SummaryWriter]:
		return None
