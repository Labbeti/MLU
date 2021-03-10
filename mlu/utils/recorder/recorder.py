
import copy

from mlu.metrics.base import IncrementalMetric
from mlu.metrics.incremental import IncrementalMean, MaxTracker
from mlu.utils.recorder.base import RecorderABC
from mlu.utils.recorder.utils import _add_subname_suffix, _add_suffixes
from torch import Tensor
from torch.utils.tensorboard.writer import SummaryWriter
from typing import Dict, Optional, Union


class Recorder(RecorderABC):
	def __init__(
		self,
		writer: Optional[SummaryWriter] = None,
		write_iteration: bool = True,
		write_epoch: bool = True,
		write_global: bool = False,
		it_suffix: str = "it",
		it_start: int = 0,
		default_epoch_trackers: Optional[Dict[str, IncrementalMetric]] = None,
		default_global_trackers: Optional[Dict[str, IncrementalMetric]] = None,
	):
		"""
			Wrapper of SummaryWriter tensorboard object.
			Useful for record epoch mean results and max of mean results.

			:param writer: SummaryWriter tensorboard object.
				(default: None)
			:param write_iteration: If True, save iteration scalar to tensorboard writer.
				(default: False)
			:param write_epoch: If True, save epoch values to tensorboard writer.
				(default: True)
			:param write_global: If True, save global value to tensorboard writer.
				(default: False)
			:param it_suffix: Iteration suffix name.
				(default: 'it')
			:param it_start: The iteration start index. Useful when the recorder is used for continue a training process.
				(default: 0)
			:param default_epoch_trackers: Default epoch tracker for compute an epoch value with scalar stored.
				If None, a IncrementalMean is used.
				(default: None)
			:param default_global_trackers: Default global trackers for compute the best values of epochs.
				If None, a MaxTracker is used.
				(default: None)
		"""
		if default_epoch_trackers is None:
			default_epoch_trackers = {"mean": IncrementalMean()}

		if default_global_trackers is None:
			default_global_trackers = {"max": MaxTracker()}

		super().__init__()
		self._writer = writer
		self._write_iteration = write_iteration
		self._write_epoch = write_epoch
		self._write_global = write_global
		self._it_suffix = it_suffix
		self._it_start = it_start
		self._default_step_trackers = default_epoch_trackers
		self._default_global_trackers = default_global_trackers

		self._idx_suffix = "idx"

		self._step_trackers = {}
		# [epoch][scalar_name][step_incr_name] -> IncrementalMean
		self._global_trackers = {}
		# [scalar_name][step_incr_name][global_incr_name] -> MaxTracker
		self._it_indexes = {}
		# [scalar_name] -> int

	def add_scalar(self, scalar_name: str, scalar: Union[float, Tensor], iteration: int, epoch: int):
		if isinstance(scalar, Tensor):
			try:
				scalar = scalar.item()
			except ValueError:
				raise ValueError(f"Cannot add to MLU Recorder a non-scalar tensor '{scalar}' (scalar_name='{scalar_name}').")

		if epoch not in self._step_trackers.keys():
			self._step_trackers[epoch] = {}
		if scalar_name not in self._step_trackers[epoch].keys():
			self._step_trackers[epoch][scalar_name] = copy.deepcopy(self._default_step_trackers)

		trackers = self._step_trackers[epoch][scalar_name]
		for step_incr_name, tracker in trackers.items():
			tracker.add(scalar)

		if self._write_iteration:
			tag = _add_subname_suffix(scalar_name, self._it_suffix)
			if tag not in self._it_indexes.keys():
				self._it_indexes[tag] = self._it_start
			else:
				self._it_indexes[tag] += 1
			self._add_scalar_to_writer(tag, scalar, self._it_indexes[tag])

	def step(self):
		for epoch, all_trackers in self._step_trackers.items():
			for scalar_name, trackers in all_trackers.items():
				for step_incr_name, tracker in trackers.items():
					tag = _add_subname_suffix(scalar_name, step_incr_name)
					if self._write_epoch:
						self._add_scalar_to_writer(tag, tracker.get_current(), epoch)

					if scalar_name not in self._global_trackers.keys():
						self._global_trackers[scalar_name] = {}
					if step_incr_name not in self._global_trackers[scalar_name].keys():
						self._global_trackers[scalar_name][step_incr_name] = copy.deepcopy(self._default_global_trackers)

					global_trackers = self._global_trackers[scalar_name][step_incr_name]

					for global_incr_name, global_tracker in global_trackers.items():
						# Update global tracker
						global_tracker.add(tracker.get_current())
						if self._write_epoch:
							# Save best data to writer
							tag = _add_suffixes(scalar_name, global_incr_name, step_incr_name)
							self._add_scalar_to_writer(tag, global_tracker.get_current(), epoch)
		self._step_trackers.clear()

	def get_current(self, epoch: int, step_incr_name: str = "mean", add_step_incr_name: bool = False) -> Dict[str, float]:
		if epoch not in self._step_trackers.keys():
			raise RuntimeError(f"Unknown epoch '{epoch}' for MLU Recorder.")

		currents = {}
		all_trackers = self._step_trackers[epoch]

		for scalar_name, trackers in all_trackers.items():
			if step_incr_name not in trackers.keys():
				raise RuntimeError(f"Unknown incremental '{step_incr_name}' at epoch '{epoch}' for metric '{scalar_name}'.")
			if add_step_incr_name:
				tag = _add_subname_suffix(scalar_name, step_incr_name)
			else:
				tag = scalar_name
			currents[tag] = trackers[step_incr_name].get_current().item()

		return currents

	def get_bests_epochs(self) -> Dict[str, Dict[str, Union[int, float]]]:
		bests = {}
		for scalar_name, all_trackers in self._global_trackers.items():
			for step_incr_name, trackers in all_trackers.items():
				tag = _add_subname_suffix(scalar_name, step_incr_name)
				bests[tag] = {}
				for global_incr_name, tracker in trackers.items():
					bests[tag][global_incr_name] = tracker.get_current().item()

					if hasattr(tracker, "get_index") and callable(tracker.get_index_best):
						index = tracker.get_index_best()
						if isinstance(index, int):
							bests[tag][f"{global_incr_name}_{self._idx_suffix}"] = index
		return bests

	def get_writer(self) -> Optional[SummaryWriter]:
		return self._writer

	def set_write_modes(self, write_iteration: bool, write_epoch: bool, write_global: bool):
		self._write_iteration = write_iteration
		self._write_epoch = write_epoch
		self._write_global = write_global

	def _add_scalar_to_writer(self, tag: str, scalar: float, step: int):
		if self._writer is not None:
			self._writer.add_scalar(tag, scalar, step)
