
import math

from torch.optim.lr_scheduler import LambdaLR
from torch.optim.optimizer import Optimizer


class CosineLRScheduler(LambdaLR):
	def __init__(self, optim: Optimizer, nb_steps: int, coefficient: float = 7.0 / 16.0):
		"""
			Scheduler that decreases the learning rate from lr0 to almost 0 by using the following rule :

			lr = lr0 * cos(7 * pi * epoch / (16 * nb_steps))

			Note : If the step() method is called more than nb_steps times, the lr will not be updated anymore.

			:param optim: The optimizer to update.
			:param nb_steps: The number of step() method call. Can be the number of epochs or iteration.
			:param coefficient: The coefficient in [0, 0.5] for controlling the decrease cosine rate.
				If closer to 0.5, the final lr will be close to 0.0
		"""
		self.nb_steps = nb_steps
		self.coefficient = coefficient
		super().__init__(optim, self.lr_lambda)

	def lr_lambda(self, step: int) -> float:
		return math.cos(self.coefficient * math.pi * min(step / self.nb_steps, 1.0))


class SoftCosineLRScheduler(LambdaLR):
	def __init__(self, optim: Optimizer, nb_steps: int):
		"""
			Scheduler that decreases the learning rate from lr0 to almost 0 by using the following rule :
			lr = lr0 * (1 + np.cos((epoch - 1) * pi / nb_epochs)) * 0.5

			:param optim: The optimizer to update.
			:param nb_steps: The number of step() method call. Can be the number of epochs or iteration.
		"""
		self.nb_steps = nb_steps
		super().__init__(optim, self.lr_lambda)

	def lr_lambda(self, step: int) -> float:
		return (1.0 + math.cos((step - 1) * math.pi / self.nb_steps)) * 0.5
