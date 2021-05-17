
import torch

from torch import Tensor
from torch.nn import Module
from torch.nn.functional import softplus
from typing import Callable


def identity(x: Tensor) -> Tensor:
	return x


def batchmean(x: Tensor) -> Tensor:
	return torch.mean(x, dim=-1)


def get_reduction_from_name(name: str) -> Callable[[Tensor], Tensor]:
	"""
		:param name: The name of the reduction function.
			Available functions are 'sum' and 'mean', 'none' and 'batchmean'.
		:return: The reduction function with a name.
	"""
	if name in ['mean']:
		return torch.mean
	elif name in ['sum']:
		return torch.sum
	elif name in ['none', 'identity']:
		return identity
	elif name in ['batchmean']:
		return batchmean
	else:
		raise RuntimeError(f'Unknown reduction "{name}". Must be one of {str(["mean", "sum", "none", "batchmean"])}.')


def get_module_checksum(model: Module, only_trainable: bool = True) -> Tensor:
	params = (param for param in model.parameters() if not only_trainable or param.requires_grad)
	return sum(param.sum() for param in params)


def get_n_parameters(model: Module, only_trainable: bool = True) -> int:
	"""
		Return the number of parameters in a module.

		:param model: Pytorch Module to check.
		:param only_trainable: If True, count only parameters that requires gradient. (default: True)
		:returns: The number of parameters.
	"""
	params = (param for param in model.parameters() if not only_trainable or param.requires_grad)
	return sum(param.numel() for param in params)


def check_module_shapes(m1: Module, m2: Module, only_trainable: bool = True) -> bool:
	params1 = [param for param in m1.parameters() if not only_trainable or param.requires_grad]
	params2 = [param for param in m2.parameters() if not only_trainable or param.requires_grad]

	return len(params1) == len(params2) and all(p1.shape == p2.shape for p1, p2 in zip(params1, params2))


def check_module_equal(m1: Module, m2: Module, only_trainable: bool = True) -> bool:
	params1 = [param for param in m1.parameters() if not only_trainable or param.requires_grad]
	params2 = [param for param in m2.parameters() if not only_trainable or param.requires_grad]

	return (
		len(params1) == len(params2) and all([p1.shape == p2.shape and p1.eq(p2).all() for p1, p2 in zip(params1, params2)])
	)


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
