
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
		dataset_params = dict(
			root=root,
			transform=Squeeze(),
			target_transform=MultiHot(200),
			download=False,
			verbose=2,
		)

		dataset_train = FSD50K(subset='train', **dataset_params)
		dataset_val = FSD50K(subset='val', **dataset_params)
		dataset_eval = FSD50K(subset='eval', **dataset_params)
		dataset_dev = FSD50K(subset='dev', **dataset_params)

		print('Dataset sizes : ')
		for dataset in (dataset_train, dataset_val, dataset_eval, dataset_dev):
			print(f'{dataset._subset} : {len(dataset)}')
		print(f'Check : {len(dataset_train) + len(dataset_val) + len(dataset_eval)}')
		print(f'Check : {len(dataset_train) + len(dataset_val)} == {len(dataset_dev)}')

		dataset = dataset_dev
		item = dataset[0]
		self.assertEqual(len(item), 2)
		audio, target = item
		self.assertTrue(isinstance(audio, Tensor))
		print(audio.shape)
		print(target.shape)

		max_n_channels = 0
		max_size = 0
		idx_max = None
		for i in tqdm.trange(len(dataset)):
			audio = dataset.get_audio(i)
			assert len(audio.shape) == 2
			if max_size < audio.shape[-1]:
				idx_max = i
			max_n_channels = max(max_n_channels, audio.shape[0])
			max_size = max(max_size, audio.shape[-1])

		print(idx_max)
		print(dataset.get_audio_fpath(idx_max))
		print(max_size)
		print(max_size / dataset.SAMPLE_RATE)
		print(max_n_channels)
		# breakpoint()


if __name__ == '__main__':
	unittest.main()
