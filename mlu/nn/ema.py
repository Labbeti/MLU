
import copy

from torch.nn import Module


class EMA(Module):
	def __init__(self, model: Module, decay: float = 0.99, copy_model: bool = True):
		"""
			Compute the exponential moving average of a model.

			>>> 'model = decay * model + (1 - decay) * other_model'

			:param model: The target model to update.
			:param decay: The exponential decay (sometimes called "alpha") used to update the model. (default: 0.99)
			:param copy_model: If true, the model passed as input will be copied. (default: True)
		"""
		super().__init__()
		if copy_model:
			model = copy.deepcopy(model)
		self.model = model
		self.decay = decay

	def update(self, other_model: Module):
		model_params = [param for param in self.model.parameters() if param.requires_grad]
		other_params = [param for param in other_model.parameters() if param.requires_grad]

		assert len(model_params) == len(other_params)

		for param, other_param in zip(model_params, other_params):
			param.set_(self.decay * param + (1.0 - self.decay) * other_param)

	def forward(self, *args, **kwargs):
		return self.model(*args, **kwargs)

	def set_model(self, model: Module):
		self.model = model

	def set_decay(self, decay: float):
		self.decay = decay

	def get_model(self) -> Module:
		return self.model

	def get_decay(self) -> float:
		return self.decay
