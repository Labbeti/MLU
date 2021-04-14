
import os.path as osp
import torch

from matplotlib import pyplot as plt
from mlu.transforms.image import RandAugment
from mlu.transforms.image.rand_augment import RAND_AUGMENT_DEFAULT_POOL
from mlu.transforms.converters import ToTensor
from torchvision.datasets import CIFAR10
from unittest import main, TestCase


class TestRandAugment(TestCase):
	def test_ra(self):
		dataset = CIFAR10(root=osp.join('../..', 'data', 'CIFAR10'), train=False, download=True, transform=None)
		idx = torch.randint(low=0, high=len(dataset), size=()).item()  # 908
		img, label = dataset[idx]

		print(f'Image index: {idx}.')
		print(f'Label: {label}.')

		magnitude = 1.0
		ra = RandAugment(n_augm_apply=1, magnitude=magnitude, augm_pool=RAND_AUGMENT_DEFAULT_POOL[2:3], magnitude_policy='constant')
		img_ra = ra(img)

		img_tens = ToTensor()(img)
		img_ra_tens = ToTensor()(img_ra)

		print(img_tens[0:5, 0, 0])
		print(img_ra_tens[0:5, 0, 0])

		print('Img    : ', img_tens.shape)
		print('Img RA : ', img_ra_tens.shape)

		if magnitude == 0:
			self.assertTrue(img_tens.eq(img_ra_tens).all())
		else:
			self.assertTrue(img_tens.ne(img_ra_tens).any())

		plt.figure()
		plt.imshow(img)

		plt.figure()
		plt.imshow(img_ra)

		plt.show(block=False)

		# input('> ')


if __name__ == '__main__':
	main()
