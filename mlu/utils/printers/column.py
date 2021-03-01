
from mlu.utils.printers.base import PrinterABC
from time import time
from typing import Dict, List


class ColumnPrinter(PrinterABC):
	def __init__(
		self,
		print_epoch: bool = True,
		print_it: bool = True,
		print_progression_percent: bool = False,
		print_exec_time: bool = True,
	):
		"""
			Class for print current values of a training or validation by columns.

			Ex:

			>>> printer = ColumnPrinter()
			>>> printer.print_current_values({"train/accuracy": 0.89, "train/loss": 1.525}, 0, 100, 2, "train")
			... '-         train          -  accuracy  -    loss    -  took (s)  -'
			... '- Epoch   2 - It   0/100 - 8.9000e-01 - 1.5250e-00 -    0.00    -'

			:param print_epoch: TODO
			:param print_it: TODO
			:param print_progression_percent: TODO
			:param print_exec_time: Print time elapsed with the beginning of the loop (iteration == 0).
		"""
		self.print_epoch = print_epoch
		self.print_it = print_it
		self.print_progression_percent = print_progression_percent
		self.print_exec_time = print_exec_time

		self._epoch_start_time = time()
		self._all_keys = []

		self._float_format = "{:.4e}"
		self._column_len = len(self._float_format.format(0))

	def print_current_values(
		self,
		current_values: Dict[str, float],
		iteration: int,
		nb_iterations: int,
		epoch: int,
		name: str,
	):
		if iteration == 0:
			keys = list(sorted(current_values.keys()))
			if name is None:
				name = "/".join(keys[0].split("/")[:-1]) if len(keys) > 0 else ""
			keys_names = [key.split("/")[-1] for key in keys]
			self._all_keys = keys
			self._print_header(name, keys_names, nb_iterations)
			self._epoch_start_time = time()
		else:
			self._all_keys += list(set(current_values.keys()).difference(self._all_keys))

		content = []
		if self.print_epoch:
			content.append("Epoch {:3d}".format(epoch))

		if self.print_it:
			it_format = f"{{:{len(str(nb_iterations))}d}}"
			it_frac_format = f"It {it_format}/{it_format}"
			content.append(it_frac_format.format(iteration, nb_iterations))

		if self.print_progression_percent:
			progression = int(100 * (iteration + 1) / nb_iterations)
			content.append("{:3d}%".format(progression))

		content += [
			self._float_format.format(current_values[key]).center(self.get_column_len())
			if key in current_values.keys() else
			" " * self.get_column_len()
			for key in self._all_keys
		]

		if self.print_exec_time:
			content.append("{:.3f}".format(time() - self._epoch_start_time).center(self.get_column_len()))

		print("- {:s} -".format(" - ".join(content)), end="\r")

		if iteration == nb_iterations - 1:
			print()

	def _print_header(self, name: str, keys: List[str], nb_iterations: int):
		def filter_name(key_name: str) -> str:
			if len(key_name) <= self.get_column_len():
				return key_name.center(self.get_column_len())
			else:
				return key_name[:self.get_column_len()]

		it_len = len(str(nb_iterations))

		start_len = 0
		nb_prints = 0
		if self.print_epoch:
			start_len += len("Epoch EEE")
			nb_prints += 1

		if self.print_it:
			start_len += len("It " + "I" * it_len + "/" + "I" * it_len)
			nb_prints += 1

		if self.print_progression_percent:
			start_len += len("XXX%")
			nb_prints += 1

		start_len += len(" - ") * ((nb_prints - 1) if nb_prints > 0 else 0)

		content = ["{:s}".format(name.center(start_len))]
		content += [filter_name(metric_name) for metric_name in keys]
		content += ["took (s)".center(self.get_column_len())]

		print("- {:s} -".format(" - ".join(content)))

	def get_column_len(self) -> int:
		return self._column_len
