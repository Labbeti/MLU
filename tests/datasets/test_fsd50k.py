
import logging
import sys
import tqdm
import unittest

from torch import Tensor
from unittest import TestCase

from mlu.datasets.fsd50k import FSD50K
from mlu.nn import Squeeze, MultiHot


class TestFSD50K(TestCase):
	def test_item(self):
		logging.basicConfig(stream=sys.stdout, level=logging.INFO)
		root = '/home/labbeti/Bureau/thesis/root_sslh/SSLH/data/FSD50K'
		dataset = FSD50K(
			root=root,
			subset='eval',
			transform=Squeeze(),
			target_transform=MultiHot(200),
			download=False,
			verbose=2,
		)

		item = dataset[0]
		self.assertEqual(len(item), 2)
		audio, target = item
		self.assertTrue(isinstance(audio, Tensor))
		print(audio.shape)
		print(target.shape)

		max_n_channels = 0
		max_size = 0
		for i in tqdm.trange(len(dataset)):
			audio = dataset.get_audio(i)
			assert len(audio.shape) == 2
			max_n_channels = max(max_n_channels, audio.shape[0])
			max_size = max(max_size, audio.shape[-1])

		print(max_size)
		print(max_size / dataset.SAMPLE_RATE)
		print(max_n_channels)


if __name__ == '__main__':
	unittest.main()
