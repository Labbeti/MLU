
import torch

from mlu.metrics.base import IncrementalMetric
from mlu.metrics.incremental import IncrementalMean, MaxTracker
from mlu.utils.recorder.base import RecorderABC
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from typing import Callable, Dict, List, Optional, Union


class Recorder(RecorderABC):
	def __init__(
		self,
		writer: Optional[SummaryWriter] = None,
		write_iteration: bool = True,
		iterations_reductions: Optional[Dict[str, Callable]] = None,
		epochs_reductions: Optional[Dict[str, Callable]] = None,
		verbose: int = 0,
	):
		super().__init__()
		if iterations_reductions is None:
			iterations_reductions = {"mean": torch.mean}
		if epochs_reductions is None:
			epochs_reductions = {"max": torch.max}

		self.writer = writer
		self.write_iteration = write_iteration
		self.iterations_reductions = iterations_reductions
		self.epochs_reductions = epochs_reductions
		self.verbose = verbose

		self._values = {}

	def step(self):
		""" Update epoch values with iteration values. """
		for reduce_epoch_name in self.epochs_reductions.keys():
			for reduce_iter_name in self.iterations_reductions.keys():
				global_values = self.get_values(reduce_epoch_name, reduce_iter_name)
				for name, scalar in global_values.items():
					self.writer.add_scalar(name, scalar)

		if self.write_iteration:
			for name, values in self._values.items():
				for epoch, epoch_values in values.items():
					for iteration, scalar in epoch_values.items():
						self.writer.add_scalar(name, scalar, iteration)

	def add_scalar(self, name: str, value: Union[float, Tensor], iteration: int, epoch: int):
		value = _to_float(value)

		if name not in self._values.keys():
			self._values[name] = {epoch: {}}
		elif epoch not in self._values[name].keys():
			self._values[name][epoch] = {}
		elif iteration in self._values[name][epoch].keys() and self.verbose >= 1:
			print(f"Overwrite value of scalar '{name}' at iteration '{iteration}' and epoch '{epoch}'.")

		self._values[name][epoch][iteration] = value

	def get_values(self, reduction_epoch: str = "max", reduction_iter: str = "mean") -> Dict[str, float]:
		reduce_epoch_fn = self.epochs_reductions[reduction_epoch]
		reduce_iter_fn = self.iterations_reductions[reduction_iter]

		reduced_values = {
			_add_subname_prefix(name, f"{reduction_epoch}_{reduction_iter}"): reduce_epoch_fn(
				torch.as_tensor(
					[reduce_iter_fn(epoch_values) for epoch_values in values.values()]
				)
			).item()
			for name, values in self._values.items()
		}
		return reduced_values

	def get_epoch_values(self, epoch: int, reduction: str = "mean") -> Dict[str, float]:
		epoch_values = {}
		for name, values in self._values.items():
			if epoch not in values.keys():
				raise RuntimeError(f"Cannot find epoch values for metric '{name}' at epoch '{epoch}'.")
			epochs_values = torch.as_tensor(values[epoch])
			name = _add_subname_prefix(name, reduction)
			reduce_iter_fn = self.iterations_reductions[reduction]
			epoch_values[name] = reduce_iter_fn(epochs_values).item()
		return epoch_values

	def get_iteration_values(self, iteration: int, epoch: int) -> Dict[str, float]:
		values = {name: values[epoch][iteration] for name, values in self._values.items()}
		return values


def _to_float(scalar: Union[float, Tensor]) -> float:
	if isinstance(scalar, Tensor):
		if torch.as_tensor(scalar, dtype=torch.int).prod() != 1:
			raise RuntimeError(f"Cannot add a non-scalar tensor of shape {str(scalar.shape)}.")
		return scalar.item()
	else:
		return scalar


def _tag_split(name: str) -> (str, str):
	split = name.split("/")
	if len(split) == 1:
		return "", split[0]
	elif len(split) == 2:
		return split[0], split[1]
	else:
		raise RuntimeError(f"Found more than 2 '/' in recorder tag '{name}'.")


def _add_subname_prefix(name: str, prefix: str) -> str:
	section, subname = _tag_split(name)
	return f"{section}/{prefix}_{subname}"
