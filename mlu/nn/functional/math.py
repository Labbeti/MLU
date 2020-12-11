
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


def modules_equals(m1: Module, m2: Module) -> bool:
	params_1 = list(m1.parameters())
	params_2 = list(m2.parameters())
	return len(params_1) == len(params_2) and all([p1.shape == p2.shape and p1.eq(p2).all() for p1, p2 in zip(params_1, params_2)])
