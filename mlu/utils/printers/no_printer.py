
from mlu.utils.printers.base import PrinterABC
from typing import Dict


class NoPrinter(PrinterABC):
	def print_current_values(
		self,
		current_values: Dict[str, float],
		iteration: int,
		nb_iterations: int,
		epoch: int,
		name: str,
	):
		pass
