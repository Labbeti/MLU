
from torch import Tensor
from torchaudio.transforms import Resample

from mlu.transforms.waveform.crop import Crop
from mlu.transforms.base import WaveformTransform
from mlu.transforms.utils import time_stretch_freq_domain


class PitchShift(WaveformTransform):
	def __init__(self, sr: int, n_steps: float, bins_per_octave: int = 12, p: float = 1.0):
		super().__init__(p=p)
		self.sr = sr
		self.n_steps = n_steps
		self.bins_per_octave = bins_per_octave

		self.resample = Resample()
		self.crop = Crop(0)

	def process(self, data: Tensor) -> Tensor:
		rate = 2.0 ** (-self.n_steps / self.bins_per_octave)

		self.resample.orig_freq = self.sr / rate
		self.resample.new_freq = self.sr
		self.crop.target_length = data.shape[-1]

		data = time_stretch_freq_domain(data, rate)
		data = self.resample(data)
		data = self.crop(data)

		return data
