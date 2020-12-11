
from mlu.utils.printers.base import PrinterABC
from time import time
from typing import Dict, List


class ColumnPrinter(PrinterABC):
	KEY_MAX_LENGTH = 10

	def __init__(self):
		self._epoch_start_date = time()
		self._keys = []

	def print_current_values(self, current_values: Dict[str, float], iteration: int, nb_iterations: int, epoch: int):
		if iteration == 0:
			keys = list(sorted(current_values.keys()))
			name = "/".join(keys[0].split("/")[:-1])
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
				"{:.4e}".format(current_values[key]).center(
					self.KEY_MAX_LENGTH) if key in current_values.keys() else " " * self.KEY_MAX_LENGTH
				for key in self._keys
			] + \
			["{:.2f}".format(time() - self._epoch_start_date).center(self.KEY_MAX_LENGTH)]

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
