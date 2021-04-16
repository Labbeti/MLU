
import torch
import unittest

from torch import Tensor
from unittest import TestCase

from mlu.transforms.noise import AdditiveNoise, gen_noise


class TestNoise(TestCase):
	def test_(self):
		x = torch.rand(32, 64, 3)
		snr_db = 40.0
		noise_module = AdditiveNoise(snr_db)

		noise = gen_noise(x, snr_db)
		x_perturbed = x + noise

		px = self.signal_power(x)
		pn = self.signal_power(noise)
		pxp = self.signal_power(x_perturbed)

		snr_db_result = 10.0 * torch.log10(px / pn)
		print("Snr dB: ", snr_db)
		print("Snr dB: ", snr_db_result)
		self.assertTrue(torch.allclose(torch.scalar_tensor(snr_db), snr_db_result, rtol=0.1))

	def signal_power(self, x: Tensor) -> Tensor:
		return (x ** 2).mean()


if __name__ == '__main__':
	unittest.main()
