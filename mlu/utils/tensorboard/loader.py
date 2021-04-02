
import os
import os.path as osp

from tensorboard.backend.event_processing.event_file_loader import EventFileLoader
from typing import Any, Callable, Dict, Iterable, List, Optional, Union


EVENT_FILE_PREFIX = "events.out.tfevents."
DT_FLOAT = 1
DT_STRING = 7


def search_fpath_rec(path: str, recursive: bool, fpath_pred: Optional[Callable]) -> List[str]:
	if osp.isfile(path) and (fpath_pred is None or fpath_pred(path)):
		return [path]
	elif osp.isdir(path):
		fpaths = []
		for subname in os.listdir(path):
			subpath = osp.join(path, subname)
			if osp.isfile(subpath) or (osp.isdir(subpath) and recursive):
				fpaths += search_fpath_rec(subpath, recursive, fpath_pred)
		return fpaths
	else:
		return []


class TensorboardLoader:
	def __init__(
		self,
		path: str,
		recursive: bool = True,
		skip_float: bool = False,
		skip_str: bool = True,
		verbose: int = 0
	):
		"""
			Build the loader of tensorboard event files.

			:param path: Filepath or dirpath to event files.
				If path is a directory, all sub-files presents will be read.
			:param recursive: Allows to check event files in subdirectories if path is a directory.
			:param skip_float: Skip float values when reading event files.
			:param skip_str: Skip string values when reading event files.
			:param verbose: Verbose level.
				If 0, no print will be done. If 1, some information will be print when reading files.
		"""
		super().__init__()
		self._path = path
		self._recursive = recursive
		self._skip_float = skip_float
		self._skip_str = skip_str
		self._verbose = verbose

		if isinstance(path, str):
			paths = [path]
		elif isinstance(path, Iterable):
			paths = path
		else:
			raise RuntimeError(f"Invalid type '{type(path)}' for TensorboardLoader.")

		event_paths = []
		for path in paths:
			event_paths += search_fpath_rec(path, recursive, lambda fpath: osp.basename(fpath).startswith(EVENT_FILE_PREFIX))
		self._event_fpaths = event_paths

	def load(self) -> Dict[str, Dict[str, Union[str, List[float]]]]:
		"""
			Load data from event paths.

			Ex:

			>>> {
			...		"train/acc": {
			...			"dtype": "float",
			...			"values": {
			...				0: 0.1,
			...				1: 0.2
			...			}
			...		}
			... }

			:return: A dictionary of floats and strings contained in event file, with values of each step.
		"""
		data = {}
		for event_fpath in self._event_fpaths:
			data.update(self._load_fpath(event_fpath))
		return data

	def _load_fpath(self, event_fpath: str) -> Dict[str, Dict[str, Any]]:
		"""

			:return: A dictionary of floats and strings contained in event file.
		"""
		event_file_loader = EventFileLoader(event_fpath)
		data = {}

		for event in event_file_loader.Load():
			event_values: list = event.summary.value
			step: int = event.step

			for event_value in event_values:
				tag: str = event_value.tag
				dtype: int = event_value.tensor.dtype

				if tag.startswith("_"):
					if self._verbose >= 1:
						print(f"Skip value with tag '{tag}' which begins by an underscore.")
					continue

				if dtype == DT_FLOAT:
					if self._skip_float:
						continue
					value = event_value.tensor.float_val
					value = float(str(value)[1:-1])
					dtype_str = "float"

				elif dtype == DT_STRING:
					if self._skip_str:
						continue
					tag = tag.split("/")[0]
					value = event_value.tensor.string_val
					value = str(value)[3:-2]
					dtype_str = "str"

				else:
					if self._verbose >= 1:
						print(f"WARNING: Unknown dtype '{dtype}'. Skip this event value with tag '{tag}' at step '{step}'.")
					continue

				if tag not in data.keys():
					data[tag] = {"dtype": dtype_str, "values": {}}

				if step in data[tag]["values"].keys():
					# Ignore events with same step, just check the same value and dtype
					assert data[tag]["dtype"] == dtype_str, \
						f"Duplicate event with tag '{tag}' on step '{step}' does not have the same dtype '{dtype}'."
					assert data[tag]["values"][step] == value, \
						f"Duplicate event with tag '{tag}' on step '{step}' does not have the same value '{value}'."
				else:
					data[tag]["values"][step] = value

		return data
