
import os.path as osp
import torch

from matplotlib import pyplot as plt
from mlu.transforms.image.pil import (
	Brightness, Color, Contrast, Sharpness,
	Posterize,
	Rotate,
	ShearX, ShearY,
	TranslateX, TranslateY,
	Solarize,
)
from torchvision.datasets import CIFAR10
from unittest import TestCase, main


class TestPIL(TestCase):
	def get_example(self):
		dataset = CIFAR10(root=osp.join("../..", "data", "CIFAR10"), train=False, download=True, transform=None)
		idx = torch.randint(low=0, high=len(dataset), size=()).item()  # 908
		img, label = dataset[idx]

		print(f"Image index: {idx}")
		print(f"Label: {label}")
		return img

	def plot_augms(self, augms):
		img = self.get_example()

		plt.figure()
		plt.imshow(img)

		for augm in augms:
			img_a = augm(img)

			plt.figure()
			plt.title(augm.__class__.__name__)
			plt.imshow(img_a)

		plt.show(block=False)
		input("> ")

	def test_enhance(self):
		level = -0.9
		augms = [
			Brightness(level),
			Color(level),
			Contrast(level),
			Sharpness(level),
		]
		self.plot_augms(augms)

	def test_posterize(self):
		nbs_bits = 7
		augms = [Posterize(nbs_bits)]
		self.plot_augms(augms)

	def test_rotation(self):
		angle = 0
		augms = [Rotate(angle)]
		self.plot_augms(augms)

	def test_shear(self):
		shear = 0.5
		augms = [ShearX(shear), ShearY(shear)]
		self.plot_augms(augms)

	def test_translate(self):
		delta = 1.0
		augms = [TranslateX(delta), TranslateY(delta)]
		self.plot_augms(augms)

	def test_solarize(self):
		threshold = 0
		augms = [Solarize(threshold)]
		self.plot_augms(augms)


if __name__ == "__main__":
	main()
