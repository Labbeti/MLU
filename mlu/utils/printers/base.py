
from abc import ABC
from typing import Dict


class PrinterABC(ABC):
	"""
		Abstract class for print values.
		A printer is a simple class that print current metrics and losses values in a specific format.
	"""

	def print_current_values(
		self,
		current_values: Dict[str, float],
		iteration: int,
		nb_iterations: int,
		epoch: int,
		name: str,
	):
		"""
			Print current values of the iteration.

			:param current_values: Continue average of all metrics.
			:param iteration: Current iteration number.
			:param nb_iterations: Number of iterations for the training/validation loop.
			:param epoch: Current epoch of the program.
			:param name: The name of the training or validation process.
		"""
		raise NotImplementedError("Abstract method")
