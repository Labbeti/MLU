
from mlu.utils import misc
from mlu.utils.misc import random_rect, random_cuboid, search_function_in_module
from unittest import TestCase, main


class TestMisc(TestCase):
	def test_rand_cuboid(self):
		ratios = [(0.2, 0.2), (0.3, 0.3)]
		sizes = [100, 200]

		left, right, top, down = random_rect(*sizes, *ratios)
		limits_rect = [(left, right), (top, down)]
		limits_cuboid = random_cuboid(sizes, ratios)

		diff_fn = lambda l: [right_ - left_ for left_, right_ in l]
		self.assertEqual(diff_fn(limits_rect), diff_fn(limits_cuboid))

	def test_search_func_in_module(self):
		func = search_function_in_module("search_function_in_module", misc)
		self.assertIsNotNone(func)


if __name__ == "__main__":
	main()
