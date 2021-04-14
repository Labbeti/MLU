
import torch
import unittest

from unittest import TestCase
from mlu.transforms import CutOutSpec


class TestCutOutSpec(TestCase):
	def test_cos(self):
		cutoutspec = CutOutSpec(fill_value=0.0)

		spec = torch.rand(2, 4, 10)
		spec_cut = cutoutspec(spec)

		# print(spec)
		# print(spec_cut)
		self.assertFalse(spec.eq(spec_cut).all())

	def test_cos_2(self):
		cutoutspec = CutOutSpec(fill_value=0.0, freq_scales=(0.4, 0.4), time_scales=(0.6, 0.6))
		spec = torch.ones(10, 100)
		spec_cut = cutoutspec(spec)
		self.assertFalse(spec.eq(spec_cut).all())
		self.assertGreater(spec.sum(), spec_cut.sum())
		self.assertEqual(spec_cut.eq(0.0).sum(), 10 * 0.4 * 100 * 0.6)

	def test_cos_3(self):
		spec = torch.ones(10, 100)
		scales = [0.0, 0.5, 1.0]

		for f_scale in scales:
			for t_scale in scales:
				cutoutspec = CutOutSpec(fill_value=0.0, freq_scales=(f_scale, f_scale), time_scales=(t_scale, t_scale))
				spec_cut = cutoutspec(spec)
				self.assertEqual(
					spec_cut.eq(0.0).sum(), 10 * f_scale * 100 * t_scale, f"Assert errors for scale {f_scale} and {t_scale}.")

	def test_fill_values(self):
		spec = torch.stack([
			torch.full((2, 5), 1.0),
			torch.full((2, 5), 2.0),
			torch.full((2, 5), 3.0),
		])
		cutoutspec = CutOutSpec(
			fill_value=(0.0, 1.0),
			freq_scales=(0.5, 0.5),
			time_scales=(0.5, 0.5),
			freq_dim=1,
			time_dim=2,
		)
		spec_cut = cutoutspec(spec)

		# print('Shape: ', spec.shape)
		# print(spec_cut)
		self.assertFalse(spec.eq(spec_cut).all())

	def test_not_same_across_channels(self):
		spec = torch.stack([
			torch.full((4, 6), 1.0),
			torch.full((4, 6), 2.0),
			torch.full((4, 6), 3.0),
		])
		cutoutspec = CutOutSpec(
			fill_value=(0.0, 0.0),
			freq_scales=(0.25, 0.25),
			time_scales=(0.5, 0.5),
			freq_dim=-2,
			time_dim=-1,
			same_across_channels=False,
		)
		spec_cut = cutoutspec(spec)
		print('Shape: ', spec.shape)
		print(spec_cut)
		self.assertFalse(spec.eq(spec_cut).all())


if __name__ == '__main__':
	unittest.main()
