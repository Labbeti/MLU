
import torch

from mlu.nn.modules.misc import DEFAULT_EPSILON, Mean
from mlu.nn.utils import get_reduction_from_name

from torch import Tensor
from torch.nn import Module, KLDivLoss, LogSoftmax, BCELoss, Sequential
from typing import Optional


class CrossEntropyWithVectors(Module):
	def __init__(
		self,
		reduction: str = "mean",
		dim: Optional[int] = -1,
		log_input: bool = False,
	):
		"""
			Compute Cross-Entropy between two distributions probabilities.

			Input and targets must be a batch of probabilities distributions of shape (batch_size, num_classes) tensor.
			Useful when target is not a one-hot label, like in Label-smoothing or MixMatch methods.
		"""
		super().__init__()
		self.reduce_fn = get_reduction_from_name(reduction)
		self.dim = dim
		self.log_input = log_input

	def forward(self, input_: Tensor, targets: Tensor) -> Tensor:
		"""
			Compute cross-entropy with targets.
			Input and target must be a (batch_size, num_classes) tensor.
		"""
		if not self.log_input:
			input_ = torch.log(input_)
		loss = -torch.sum(input_ * targets, dim=self.dim)
		return self.reduce_fn(loss)

	def extra_repr(self) -> str:
		return f"reduce_fn={self.reduce_fn.__name__}, dim={self.dim}, log_input={self.log_input}"


class Entropy(Module):
	def __init__(
		self,
		reduction: str = "mean",
		dim: Optional[int] = -1,
		epsilon: float = DEFAULT_EPSILON,
		base: Optional[float] = None,
		log_input: bool = False,
	):
		"""
			Compute the entropy of a distribution.

			:param reduction: The reduction used between batch entropies. (default: 'mean')
			:param dim: The dimension to apply the sum in entropy formula. (default: -1)
			:param epsilon: The epsilon precision to use. Must be a small positive float. (default: DEFAULT_EPSILON)
			:param base: The log-base used. If None, use the natural logarithm (i.e. base = torch.exp(1)). (default: None)
			:param log_input: If True, the input must be log-probabilities. (default: False)
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

	def forward(self, input_: Tensor) -> Tensor:
		if not self.log_input:
			entropy = - torch.sum(input_ * self.log_func(input_ + self.epsilon), dim=self.dim)
		else:
			entropy = - torch.sum(torch.exp(input_) * input_, dim=self.dim)
		return self.reduce_fn(entropy)

	def extra_repr(self) -> str:
		return f"reduce_fn={self.reduce_fn.__name__}, dim={self.dim}, epsilon={self.epsilon}, log_input={self.log_input}"


class JSDivLoss(Module):
	def __init__(self, reduction: str = "mean", dim: int = -1, epsilon: float = DEFAULT_EPSILON):
		"""
			Jensen-Shannon Divergence loss.

			Use the following formula :

			>>> 'JS(p,q) = H(0.5 * (p+q)) - 0.5 * (H(p) + H(q))'

			:param reduction: The reduction function to apply. (default: 'mean')
			:param dim: The dimension of the probabilities. (default: -1)
			:param epsilon: The epsilon value used for numerical stability. (default: DEFAULT_EPSILON)
		"""
		super().__init__()
		self.entropy = Entropy(reduction, dim, epsilon, log_input=False)

	def forward(self, p: Tensor, q: Tensor) -> Tensor:
		a = self.entropy(0.5 * (p + q))
		b = 0.5 * (self.entropy(p) + self.entropy(q))
		return a - b


class JSDivLossFromLogits(Module):
	def __init__(self, reduction: str = "mean", log_activation: Module = LogSoftmax(dim=-1)):
		"""
			Jensen-Shannon Divergence loss with logits.

			Use the following formula :

			>>> 'JS(p,q) = 0.5 * (KL(LS(p),m) + KL(LS(q),m)), with m = LS(0.5 * (p+q)), LS = LogSoftmax and KL = KL-Divergence.'

			:param reduction: The reduction function to apply. (default: 'mean')
			:param log_activation: The log-activation function for compute predictions from logits. (default: LogSoftmax(dim=-1))
		"""
		super().__init__()
		self.kl_div = KLDivLoss(reduction=reduction, log_target=True)
		self.log_activation = log_activation

	def forward(self, logits_p: Tensor, logits_q: Tensor) -> Tensor:
		m = self.log_activation(0.5 * (logits_p + logits_q))
		p = self.log_activation(logits_p)
		q = self.log_activation(logits_q)

		a = self.kl_div(p, m)
		b = self.kl_div(q, m)

		return 0.5 * (a + b)


class KLDivLossWithProbabilities(KLDivLoss):
	def __init__(
		self,
		reduction: str = "mean",
		epsilon: float = DEFAULT_EPSILON,
		log_input: bool = False,
		log_target: bool = False,
	):
		"""
			KL-divergence with probabilities.

			:param reduction: The reduction function to apply. (default: 'mean')
			:param epsilon: The epsilon value used for numerical stability. (default: DEFAULT_EPSILON)
			:param log_input: If True, the probabilities of the first argument are transform to log scale internally.
			:param log_target:

		"""
		super().__init__(reduction=reduction, log_target=log_target)
		self.epsilon = epsilon
		self.log_input = log_input
		self.log_target = log_target

	def forward(self, p: Tensor, q: Tensor) -> Tensor:
		if not self.log_input:
			p = torch.log(p + self.epsilon)
		return super().forward(input=p, target=q)

	def extra_repr(self) -> str:
		return f"epsilon={self.epsilon}, log_input={self.log_input}, log_target={self.log_target}"


class BCELossBatchMean(Module):
	def __init__(self, dim: Optional[int] = -1):
		super().__init__()
		self.bce = BCELoss(reduction="none")
		self.mean = Mean(dim=dim)

	def forward(self, input_: Tensor, target: Tensor) -> Tensor:
		return self.mean(self.bce(input_, target))

	@property
	def dim(self) -> int:
		return self.__getitem__(1).dim

	def extra_repr(self) -> str:
		return f"dim={self.dim}"
