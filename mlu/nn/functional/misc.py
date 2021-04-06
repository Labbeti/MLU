
import torch

from torch import Tensor
from torch.nn import Module
from torch.nn.functional import softplus


def harmonic_mean(sequence: Tensor, dim: int = 0) -> Tensor:
	return sequence.shape[dim] / ((1.0 / sequence).sum(dim=dim))


def geometric_mean(sequence: Tensor, dim: int = 0) -> Tensor:
	return sequence.prod(dim=dim) ** (1.0 / sequence.shape[dim])


def arithmetic_mean(sequence: Tensor, dim: int = 0) -> Tensor:
	return sequence.mean(dim=dim)


def mish(x: Tensor) -> Tensor:
	return x * torch.tanh(softplus(x))
