
import torch
import unittest

from unittest import TestCase
from mlu.transforms import CutOutSpec


class TestCutOutSpec(TestCase):
	def test_cos_2(self):
		cutoutspec = CutOutSpec(fill_value=0.0, freq_scales=(0.4, 0.4), time_scales=(0.6, 0.6))
		spec = torch.ones(10, 100)
		spec_cut = cutoutspec(spec)
		self.assertFalse(spec.eq(spec_cut).all())
		self.assertGreater(spec.sum(), spec_cut.sum())
		self.assertEqual(spec_cut.eq(0.0).sum(), 10 * 0.4 * 100 * 0.6)

	def test_cos(self):
		cutoutspec = CutOutSpec(fill_value=0.0)

		spec = torch.rand(2, 4, 10)
		spec_cut = cutoutspec(spec)

		# print(spec)
		# print(spec_cut)
		self.assertFalse(spec.eq(spec_cut).all())


if __name__ == '__main__':
	unittest.main()
