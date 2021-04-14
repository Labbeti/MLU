
import copy
import torch

from torch.nn import Module
from typing import Any


class EMA(Module):
	def __init__(self, model: Module, decay: float = 0.99, copy_model: bool = False):
		"""
			Compute the exponential moving average of a model.

			>>> 'model = decay * model + (1 - decay) * other_model'

			:param model: The target model to update.
			:param decay: The exponential decay (sometimes called 'alpha') used to update the model. (default: 0.99)
			:param copy_model: If True, the model passed as input will be copied. (default: False)
		"""
		if copy_model:
			model = copy.deepcopy(model)

		super().__init__()
		self.model = model
		self.decay = decay

	def update(self, other_model: Module):
		model_params = [param for param in self.model.parameters()]
		other_params = [param for param in other_model.parameters()]

		if len(model_params) != len(other_params):
			raise RuntimeError('For EMA, model used for update is supposed to have the same architecture.')

		for param, other_param in zip(model_params, other_params):
			param.set_(self.decay * param + (1.0 - self.decay) * other_param)

	def forward(self, *args, **kwargs):
		return self.model(*args, **kwargs)

	def __setattr__(self, name: str, value: Any):
		if name not in ['model']:
			Module.__setattr__(self, name, value)
		else:
			object.__setattr__(self, name, value)

	def extra_repr(self) -> str:
		return f'decay={self.decay}'
