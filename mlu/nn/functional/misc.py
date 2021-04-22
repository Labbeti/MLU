
import torch

from torch import Tensor
from torch.nn.functional import softplus


def harmonic_mean(sequence: Tensor, dim: int = 0) -> Tensor:
	return sequence.shape[dim] / ((1.0 / sequence).sum(dim=dim))


def geometric_mean(sequence: Tensor, dim: int = 0) -> Tensor:
	return sequence.prod(dim=dim) ** (1.0 / sequence.shape[dim])


def arithmetic_mean(sequence: Tensor, dim: int = 0) -> Tensor:
	return sequence.mean(dim=dim)


def mish(x: Tensor) -> Tensor:
	return x * torch.tanh(softplus(x))


def cplx_spectrogram(
	waveform: Tensor,
	pad: int,
	window: Tensor,
	n_fft: int,
	hop_length: int,
	win_length: int,
	normalized: bool,
	center: bool = True,
	pad_mode: str = "reflect",
	onesided: bool = True
) -> Tensor:
	"""
		Based on torchaudio 'spectrogram' function.
		TODO : doc
	"""

	if pad > 0:
		waveform = torch.nn.functional.pad(waveform, (pad, pad), "constant")

	# pack batch
	shape = waveform.size()
	waveform = waveform.reshape(-1, shape[-1])

	# default values are consistent with librosa.core.spectrum._spectrogram
	spec_f = torch.stft(
		input=waveform,
		n_fft=n_fft,
		hop_length=hop_length,
		win_length=win_length,
		window=window,
		center=center,
		pad_mode=pad_mode,
		normalized=False,
		onesided=onesided,
		return_complex=True,
	)

	# unpack batch
	spec_f = spec_f.reshape(shape[:-1] + spec_f.shape[-2:])

	if normalized:
		spec_f /= window.pow(2.).sum().sqrt()
	return spec_f
