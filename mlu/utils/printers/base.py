
from abc import ABC
from typing import Dict


class PrinterABC(ABC):
	"""
		Abstract class for print values.
		A printer is a simple class that print current metrics and losses values in a specific format.
	"""

	def print_current_values(self, current_means: Dict[str, float], iteration: int, nb_iterations: int, epoch: int):
		"""
			Print current values of the iteration.

			:param current_means: Continue average of all metrics.
			:param iteration: Current iteration number.
			:param nb_iterations: Number of iterations for the training/validation loop.
			:param epoch: Current epoch of the program.
		"""
		raise NotImplementedError("Abstract method")
