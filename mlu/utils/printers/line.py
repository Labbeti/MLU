
from mlu.utils.printers.base import PrinterABC
from time import time
from typing import Dict


class LinePrinter(PrinterABC):
	def __init__(self, print_exec_time: bool = True):
		self.print_exec_time = print_exec_time
		self._epoch_start_date = time()

	def print_current_values(self, current_means: Dict[str, float], iteration: int, nb_iterations: int, epoch: int):
		if iteration == 0:
			self._epoch_start_date = time()

		keys = list(current_means.keys())
		name = "/".join(keys[0].split("/")[:-1])

		progression = int(100 * (iteration + 1) / nb_iterations)
		content = ["{:s}: {:.4e}".format(name, mean) for name, mean in current_means.items()]
		if self.print_exec_time:
			content.append("{:.2f}".format(time() - self._epoch_start_date))
		content = ", ".join(content)
		print("{:5s}, epoch {:3d}, {:3d}%, {:s}".format(name, epoch, progression, content), end="\r")

		if iteration == nb_iterations - 1:
			print("")
