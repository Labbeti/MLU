
import unittest

from mlu.utils.zip_cycle import ZipCycle
from unittest import TestCase


class TestZipCycle(TestCase):
	def test(self):
		r1 = range(1, 4)
		r2 = range(1, 6)
		zip_cycle = ZipCycle(r1, r2, policy="max")

		expected = [
			(1, 1),
			(2, 2),
			(3, 3),
			(1, 4),
			(2, 5),
		]

		self.assertEqual(len(zip_cycle), len(expected))

		for i, (v1, v2) in enumerate(zip_cycle):
			e1, e2 = expected[i]
			self.assertEqual(v1, e1)
			self.assertEqual(v2, e2)


if __name__ == "__main__":
	unittest.main()
