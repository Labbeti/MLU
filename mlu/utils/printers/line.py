
from mlu.utils.printers.base import PrinterABC
from time import time
from typing import Dict


class LinePrinter(PrinterABC):
	def __init__(
		self,
		print_exec_time: bool = True,
	):
		"""
			Class for print current values of a training or validation in one line.

			Ex:
			> printer = LinePrinter()
			> printer.print_current_values({"train/accuracy": 0.89, "train/loss": 1.525}, 33, 100, 2, "train")
			train, epoch   2,  33%, accuracy: 8.9000e-01, loss: 1.5250e+00, took (s): 0.0

			:param print_exec_time: Print time elapsed with the beginning of the loop (iteration == 0).
		"""
		self.print_exec_time = print_exec_time
		self._epoch_start_date = time()

	def print_current_values(
		self,
		current_values: Dict[str, float],
		iteration: int,
		nb_iterations: int,
		epoch: int,
		name: str,
	):
		if iteration == 0:
			self._epoch_start_date = time()

		keys = list(current_values.keys())
		if name is None:
			name = "/".join(keys[0].split("/")[:-1]) if "/" in keys[0] else ""
		progression = int(100 * (iteration + 1) / nb_iterations)
		current_means = {name.split("/")[-1]: value for name, value in current_values.items()}

		content = ["{:s}: {:.4e}".format(name, value) for name, value in current_means.items()]
		if self.print_exec_time:
			content.append("{:s}: {:.2f}".format("took (s)", time() - self._epoch_start_date))
		content = ", ".join(content)
		print("{:5s}, epoch {:3d}, {:3d}%, {:s}".format(name, epoch+1, progression, content), end="\r")

		if iteration == nb_iterations - 1:
			print("")
