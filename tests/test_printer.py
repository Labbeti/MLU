
import unittest

from utils.old_ign.printers import ColumnPrinter
from unittest import TestCase


class TestPrinter(TestCase):
	def test_column_printer(self):
		printer = ColumnPrinter(print_progression_percent=False)

		printer.print_current_values({"train/accuracy": 0.89, "train/loss": 1.525}, 0, 100, 2, "train")
		print()
		print()

		nb_epoch = 1
		nb_it = 50000
		for epoch in range(nb_epoch):
			for it in range(nb_it):
				printer.print_current_values(
					{"train/acc": epoch / nb_epoch + it, "train/loss": it ** 0.5}, it, nb_it, epoch, "train"
				)


if __name__ == "__main__":
	unittest.main()
