
import numpy as np
import torch

from mlu.nn.functional.labels import nums_to_smooth_onehot, nums_to_smooth_multihot

from torch import Tensor
from torch.nn import Module
from typing import Callable, Optional, Union


class OneHot(Module):
	def __init__(self, num_classes: int, smooth: Optional[float] = 0.0):
		"""
			Convert label to one-hot encoding.

			:param num_classes: The number of classes in the dataset.
			:param smooth: The optional label smoothing coefficient parameter. (default: 0.0)
		"""
		super().__init__()
		self.num_classes = num_classes
		self.smooth = smooth if smooth is not None else 0.0

	def forward(self, x: Union[np.ndarray, Tensor]) -> Union[np.ndarray, Tensor]:
		return nums_to_smooth_onehot(x, self.num_classes, self.smooth)


class MultiHot(Module):
	def __init__(self, num_classes: int, smooth: Optional[float] = 0.0, dtype: torch.dtype = torch.float):
		super().__init__()
		self.num_classes = num_classes
		self.smooth = smooth if smooth is not None else 0.0
		self.dtype = dtype

	def forward(self, x: Union[np.ndarray, Tensor]) -> Union[np.ndarray, Tensor]:
		return nums_to_smooth_multihot(x, self.num_classes, self.smooth, self.dtype)


class Thresholding(Module):
	def __init__(self, threshold: Optional[float], bin_func: Callable = torch.ge):
		"""
			Convert label to multi-hot encoding.

			:param threshold: The threshold used to binarize the input.
				If None, the forward of this module will have no effect.
			:param bin_func: The comparison function used to binarize the Tensor. (default: torch.ge)
		"""
		super().__init__()
		self.threshold = threshold
		self.bin_func = bin_func

	def forward(self, x: Tensor) -> Tensor:
		if self.threshold is not None:
			return self.bin_func(x, self.threshold).to(x.dtype)
		else:
			return x
