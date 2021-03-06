
from mlu.utils.printers.base import PrinterABC
from time import time
from typing import Dict


class LinePrinter(PrinterABC):
	def __init__(
		self,
		print_epoch: bool = True,
		print_it: bool = True,
		print_progression_percent: bool = False,
		print_exec_time: bool = True,
	):
		"""
			Class for print current values of a training or validation in one line.

			Ex:

			>>> printer = LinePrinter()
			>>> printer.print_current_values({"train/accuracy": 0.89, "train/loss": 1.525}, 33, 100, 2, "train")
			... 'train, epoch   2, it  33/100, accuracy: 8.9000e-01, loss: 1.5250e+00, took (s): 0.0, it: 33/100'

			:param print_epoch: TODO
				(default: True)
			:param print_it: Print the iteration over the max number of iterations.
				(default: False)
			:param print_progression_percent: TODO
				(default: False)
			:param print_exec_time: Print time elapsed with the beginning of the loop (iteration == 0).
				(default: True)
		"""
		self.print_epoch = print_epoch
		self.print_it = print_it
		self.print_progression_percent = print_progression_percent
		self.print_exec_time = print_exec_time

		self._epoch_start_time = time()

	def print_current_values(
		self,
		current_values: Dict[str, float],
		iteration: int,
		nb_iterations: int,
		epoch: int,
		name: str,
	):
		if iteration == 0:
			self._epoch_start_time = time()

		if name is None:
			keys = list(current_values.keys())
			name = "/".join(keys[0].split("/")[:-1]) if "/" in keys[0] else ""

		content = ["{:5s}".format(name)]

		if self.print_epoch:
			content.append("epoch {:3d}".format(epoch))

		if self.print_it:
			it_format = f"{{:{len(str(nb_iterations))}d}}"
			it_frac_format = f"it {it_format}/{it_format}"
			content.append(it_frac_format.format(iteration, nb_iterations))

		if self.print_progression_percent:
			progression = int(100 * (iteration + 1) / nb_iterations)
			content.append("{:3d}%".format(progression))

		current_means = {name.split("/")[-1]: value for name, value in current_values.items()}
		content += ["{:s}: {:.4e}".format(name, value) for name, value in current_means.items()]

		if self.print_exec_time:
			content.append("{:s}: {:.3f}".format("took (s)", time() - self._epoch_start_time))

		content = ", ".join(content)
		print(content, end="\r")

		if iteration >= nb_iterations:
			print()
