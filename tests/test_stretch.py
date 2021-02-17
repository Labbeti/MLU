
import math
import torch

from mlu.transforms.waveform import StretchNearestFreq, StretchNearestRate
from mlu.transforms.waveform.stretch_pad_crop import StretchPadCrop, StretchPadCrop2
from unittest import TestCase, main


class TestStretch(TestCase):
	def test_stretch(self):
		t1 = StretchNearestFreq(orig_freq=1, new_freq=2)
		t2 = StretchNearestRate(rates=2)

		waveform = torch.as_tensor(list(range(10)))

		w1 = t1(waveform)
		w2 = t2(waveform)

		expected = torch.as_tensor([math.floor(i/2) for i in range(20)], dtype=torch.float)
		self.assertTrue(w1.eq(expected).all())
		self.assertTrue(w1.eq(w2).all())

	def test_stretch_pad_crop(self):
		t1 = StretchPadCrop(rates=0.5, align="left")
		t2 = StretchPadCrop2(rates=0.5, align="left")

		waveform = torch.as_tensor(list(range(10)))

		w1 = t1(waveform)
		w2 = t2(waveform)

		expected = torch.as_tensor([0, 2, 4, 6, 8, 0, 0, 0, 0, 0], dtype=torch.float)
		self.assertTrue(w1.eq(expected).all())
		self.assertTrue(w1.eq(w2).all())


if __name__ == "__main__":
	main()
