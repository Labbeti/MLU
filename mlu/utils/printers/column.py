
from mlu.utils.printers.base import PrinterABC
from time import time
from typing import Dict, List


class ColumnPrinter(PrinterABC):
	KEY_MAX_LENGTH = 10

	def __init__(self, print_exec_time: bool = True):
		"""
			Class for print current values of a training or validation by columns.

			Ex:
			> printer = ColumnPrinter()
			> printer.print_current_values({"train/accuracy": 0.89, "train/loss": 1.525}, 33, 100, 2)
			-      train       -  accuracy  -    loss    -  took (s)  -
			- Epoch   2 -  33% - 8.9000e-01 - 1.5250e-00 -    0.00    -

			:param print_exec_time: Print time elapsed with the beginning of the loop (iteration == 0).
		"""
		self.print_exec_time = print_exec_time

		self._epoch_start_date = time()
		self._keys = []

	def print_current_values(self, current_values: Dict[str, float], iteration: int, nb_iterations: int, epoch: int):
		if iteration == 0:
			keys = list(sorted(current_values.keys()))
			name = "/".join(keys[0].split("/")[:-1]) if len(keys) > 0 else ""
			keys_names = [key.split("/")[-1] for key in keys]
			self._keys = keys
			self._print_header(name, keys_names)
			self._epoch_start_date = time()
		else:
			self._keys += list(set(current_values.keys()).difference(self._keys))

		progression = int(100 * (iteration + 1) / nb_iterations)
		content = \
			["Epoch {:3d} - {:3d}%".format(epoch + 1, progression)] + \
			[
				"{:.4e}".format(current_values[key]).center(self.KEY_MAX_LENGTH)
				if key in current_values.keys() else
				" " * self.KEY_MAX_LENGTH
				for key in self._keys
			]

		if self.print_exec_time:
			content += ["{:.2f}".format(time() - self._epoch_start_date).center(self.KEY_MAX_LENGTH)]

		print("- {:s} -".format(" - ".join(content)), end="\r")

		if iteration == nb_iterations - 1:
			print("")

	def _print_header(self, name: str, keys: List[str]):
		def filter_name(key_name: str) -> str:
			if len(key_name) <= self.KEY_MAX_LENGTH:
				return key_name.center(self.KEY_MAX_LENGTH)
			else:
				return key_name[:self.KEY_MAX_LENGTH]

		content = ["{:s}".format(name.center(16))]
		content += [filter_name(metric_name) for metric_name in keys]
		content += ["took (s)".center(self.KEY_MAX_LENGTH)]

		print("- {:s} -".format(" - ".join(content)))
