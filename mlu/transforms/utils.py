
import math
import torch

from torch import Tensor
from torch.nn.functional import pad
from typing import Optional

from mlu.transforms.base import Transform


class Identity(Transform):
	def __init__(self, p: float = 1.0):
		"""
			Identity transform.

			:param p: The probability to apply the transform. (default: 1.0)
		"""
		super().__init__(p=p)

	def process(self, x: Tensor) -> Tensor:
		return x

	def is_image_transform(self) -> bool:
		return True

	def is_waveform_transform(self) -> bool:
		return True

	def is_spectrogram_transform(self) -> bool:
		return True


def default_extra_repr(obj: object, skip_private_attr: bool = True) -> str:
	attributes = [
		f'name={value}' for name, value in obj.__dict__.items() if not skip_private_attr or not name.startswith('_')
	]
	return ', '.join(attributes)


def phase_vocoder(data: Tensor, rate: float, hop_length: Optional[int] = None) -> Tensor:
	"""
		Based on librosa implementation :
		https://librosa.org/doc/main/_modules/librosa/core/spectrum.html#phase_vocoder
	"""
	assert len(data.shape) >= 2

	device = data.device
	n_fft = 2 * (data.shape[0] - 1)

	if hop_length is None:
		hop_length = int(n_fft // 4)

	time_steps = torch.arange(0, data.shape[-1], rate, dtype=torch.float, device=device)

	# Create an empty output array
	d_stretch = torch.zeros(*data.shape[:-1], len(time_steps), dtype=data.dtype, device=device)

	# Expected phase advance in each bin
	phi_advance = torch.linspace(0, math.pi * hop_length, data.shape[-2], device=device)

	# Phase accumulator; initialize to the first sample
	phase_acc = torch.angle(data[:, 0])

	# Pad 0 columns to simplify boundary logic
	data = pad(data, [0, 2, 0, 0], mode='constant')

	for (t, step) in enumerate(time_steps):
		slices = [slice(None)] * len(data.shape)
		slices[-1] = slice(int(step), int(step + 2))
		columns = data[slices]

		# Weighting for linear magnitude interpolation
		alpha = torch.fmod(step, 1.0)
		mag = (1.0 - alpha) * torch.abs(columns[:, 0]) + alpha * torch.abs(columns[:, 1])

		# Store to output array
		print("d_stretch =", d_stretch.shape)
		print("d_stretch[:, t] =", d_stretch[:, t].shape)
		print("mag =", mag.shape)
		print("phase_acc =", phase_acc.shape)
		d_stretch[:, t] = mag * torch.as_tensor(1.j * phase_acc).exp()

		# Compute phase advance
		dphase = torch.angle(columns[:, 1]) - torch.angle(columns[:, 0]) - phi_advance

		# Wrap to -pi:pi range
		dphase = dphase - 2.0 * math.pi * torch.round(dphase / (2.0 * math.pi))

		# Accumulate phase
		phase_acc += phi_advance + dphase

	return d_stretch


def time_stretch_freq_domain(data: Tensor, rate: float) -> Tensor:
	"""
		Based on librosa implementation :
		https://man.hubwiz.com/docset/LibROSA.docset/Contents/Resources/Documents/_modules/librosa/effects.html#time_stretch
	"""
	stft = torch.stft(data, n_fft=2048, return_complex=True)
	stft_stretch = phase_vocoder(stft, rate)
	data_stretch = torch.istft(stft_stretch, n_fft=2048)
	return data_stretch
