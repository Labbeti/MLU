
import torch
import unittest

from unittest import TestCase
from mlu.transforms.waveform import Occlusion


class TestOcclusion(TestCase):
	def test_occlusion(self):
		p = 1.0
		occlusion = Occlusion(0.1, p=p)
		waveform = torch.ones(100)

		waveform_perturbed = occlusion(waveform)

		self.assertEqual(waveform.sum(), 100)
		self.assertEqual(waveform_perturbed.sum(), 90)


if __name__ == '__main__':
	unittest.main()
