
import torch

from mlu.transforms.waveform import StretchNearestFreq, StretchNearestRate
from unittest import TestCase, main


class TestStretch(TestCase):
	def test_stretch(self):
		t1 = StretchNearestFreq(orig_freq=1, new_freq=2)
		t2 = StretchNearestRate(rates=2)

		waveform = torch.as_tensor(list(range(10)))

		w1 = t1(waveform)
		w2 = t2(waveform)

		self.assertTrue(w1.eq(w2).all())


if __name__ == "__main__":
	main()
