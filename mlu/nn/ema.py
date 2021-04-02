
import copy

from torch.nn import Module, Parameter
from typing import Iterator


class EMA(Module):
	def __init__(self, model: Module, decay: float = 0.99, copy_model: bool = False):
		"""
			Compute the exponential moving average of a model.

			>>> 'model = decay * model + (1 - decay) * other_model'

			:param model: The target model to update.
			:param decay: The exponential decay (sometimes called "alpha") used to update the model. (default: 0.99)
			:param copy_model: If True, the model passed as input will be copied. (default: False)
		"""
		super().__init__()
		if copy_model:
			model = copy.deepcopy(model)
		self.model = model
		self.decay = decay

	def update(self, other_model: Module):
		model_params = [param for param in self.model.parameters()]
		other_params = [param for param in other_model.parameters()]

		assert len(model_params) == len(other_params), \
			"For EMA, models used for update is supposed to have the same architecture."

		for param, other_param in zip(model_params, other_params):
			param.set_(self.decay * param + (1.0 - self.decay) * other_param)

	def forward(self, *args, **kwargs):
		return self.model(*args, **kwargs)
