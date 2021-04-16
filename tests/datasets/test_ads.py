
import unittest

from mlu.datasets import AudioSet
from unittest import TestCase


class LoadTest(TestCase):
	def test_load_first(self):
		dataset_root = '/projets/samova/CORPORA/AUDIOSET/'
		dataset = AudioSet(dataset_root, 'eval', verbose=0)

		print(len(dataset))
		print(dataset[0])
		print(dataset[0][0].shape)
		print(dataset[0][1].shape)


if __name__ == '__main__':
	unittest.main()
