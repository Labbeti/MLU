
import math

from torch.optim.lr_scheduler import LambdaLR
from torch.optim.optimizer import Optimizer


class CosineLRScheduler(LambdaLR):
	def __init__(self, optimizer: Optimizer, n_steps: int, coefficient: float = 7.0 / 16.0):
		"""
			Scheduler that decreases the learning rate from lr0 to almost 0 by using the following rule :

			>>> 'lr = lr0 * cos(7 * pi * epoch / (16 * n_steps))'

			Note : Used in FixMatch method.
			Note : If the step() method is called more than n_steps times, the lr will not be updated anymore.

			:param optimizer: The optimizer to update.
			:param n_steps: The number of step() method call. Can be the number of epochs or iteration.
			:param coefficient: The coefficient in [0, 0.5] for controlling the decrease cosine rate.
				If closer to 0.5, the final lr will be close to 0.0
		"""
		self.n_steps = n_steps
		self.coefficient = coefficient
		super().__init__(optimizer, self.lr_lambda)

	def lr_lambda(self, step: int) -> float:
		return math.cos(self.coefficient * math.pi * min(step / self.n_steps, 1.0))


class SoftCosineLRScheduler(LambdaLR):
	def __init__(self, optimizer: Optimizer, n_steps: int):
		"""
			Scheduler that decreases the learning rate from lr0 to almost 0 by using the following rule :

			>>> 'lr = lr0 * (1 + np.cos((epoch - 1) * pi / n_epochs)) * 0.5'

			:param optimizer: The optimizer to update.
			:param n_steps: The number of step() method call. Can be the number of epochs or iteration.
		"""
		self.n_steps = n_steps
		super().__init__(optimizer, self.lr_lambda)

	def lr_lambda(self, step: int) -> float:
		return (1.0 + math.cos((step - 1) * math.pi / self.n_steps)) * 0.5
