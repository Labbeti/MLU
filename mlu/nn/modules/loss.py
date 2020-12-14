
import torch

from mlu.nn.modules.math import DEFAULT_EPSILON
from mlu.nn.utils import get_reduction_from_name

from torch import Tensor
from torch.nn import Module, KLDivLoss, LogSoftmax
from typing import Optional


class Entropy(Module):
	def __init__(
		self,
		reduction: str = "mean",
		dim: int = 1,
		epsilon: float = DEFAULT_EPSILON,
		base: Optional[float] = None,
		log_input: bool = False,
	):
		"""
			Compute the entropy of a distribution.

			:param reduction: The reduction used between batch entropies.
			:param dim: The dimension to apply the sum in entropy formula.
			:param epsilon: The epsilon precision to use. Must be a small positive float.
			:param base: The log-base used. If None, use the natural logarithm (i.e. base = torch.exp(1)).
		"""
		super().__init__()
		self.reduce_fn = get_reduction_from_name(reduction)
		self.dim = dim
		self.epsilon = epsilon
		self.log_input = log_input

		if base is None:
			self.log_func = torch.log
		else:
			log_base = torch.log(torch.scalar_tensor(base))
			self.log_func = lambda x: torch.log(x) / log_base

	def forward(self, input_: Tensor, dim: Optional[int] = None) -> Tensor:
		if dim is None:
			dim = self.dim
		if not self.log_input:
			entropy = - torch.sum(input_ * self.log_func(input_ + self.epsilon), dim=dim)
		else:
			entropy = - torch.sum(torch.exp(input_) * input_, dim=dim)
		return self.reduce_fn(entropy)


class CrossEntropyWithVectors(Module):
	"""
		Compute Cross-Entropy between two distributions.
		Input and targets must be a batch of probabilities distributions of shape (batch_size, nb_classes) tensor.
	"""
	def __init__(self, reduction: str = "mean", dim: Optional[int] = 1, log_input: bool = False):
		super().__init__()
		self.reduce_fn = get_reduction_from_name(reduction)
		self.dim = dim
		self.log_input = log_input

	def forward(self, input_: Tensor, targets: Tensor, dim: Optional[int] = None) -> Tensor:
		"""
			Compute cross-entropy with targets.
			Input and target must be a (batch_size, nb_classes) tensor.
		"""
		if dim is None:
			dim = self.dim
		if not self.log_input:
			input_ = torch.log(input_)
		loss = -torch.sum(input_ * targets, dim=dim)
		return self.reduce_fn(loss)


class KLDivLossWithProbabilities(KLDivLoss):
	"""
		KL divergence with probabilities.
		The probabilities are transform to log scale internally.
	"""
	def __init__(self, reduction: str = "mean", epsilon: float = DEFAULT_EPSILON, log_input: bool = False, log_target: bool = False):
		super().__init__(reduction=reduction, log_target=True)
		self.epsilon = epsilon
		self.log_input = log_input
		self.log_target = log_target

	def forward(self, p: Tensor, q: Tensor) -> Tensor:
		if not self.log_input:
			p = torch.log(p + self.epsilon)
		if not self.log_target:
			q = torch.log(q + self.epsilon)
		return super().forward(input=p, target=q)


class JSDivLoss(Module):
	"""
		Jensen-Shannon Divergence loss.
		Use Entropy as backend.
	"""

	def __init__(self, reduction: str = "mean", dim: int = 1, epsilon: float = DEFAULT_EPSILON):
		super().__init__()
		self.entropy = Entropy(reduction, dim, epsilon)

	def forward(self, p: Tensor, q: Tensor) -> Tensor:
		a = self.entropy(0.5 * (p + q))
		b = 0.5 * (self.entropy(p) + self.entropy(q))
		return a - b


class JSDivLossWithLogits(Module):
	"""
		Jensen-Shannon Divergence loss with logits.
		Use KLDivLoss and LogSoftmax as backend.
	"""

	def __init__(self, reduction: str = "mean"):
		super().__init__()
		self.kl_div = KLDivLoss(reduction=reduction, log_target=True)
		self.log_softmax = LogSoftmax(dim=1)

	def forward(self, logits_p: Tensor, logits_q: Tensor):
		m = self.log_softmax(0.5 * (logits_p + logits_q))
		p = self.log_softmax(logits_p)
		q = self.log_softmax(logits_q)

		a = self.kl_div(p, m)
		b = self.kl_div(q, m)

		return 0.5 * (a + b)
