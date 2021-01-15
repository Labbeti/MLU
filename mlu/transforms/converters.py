
import numpy as np
import torch
import torchvision as tv

from PIL import Image
from torch import Tensor
from torch.nn import Module
from typing import Optional, Union


class ToNumpy(Module):
	def __init__(self, dtype: Optional[object] = None):
		"""
			Convert a python list, pytorch tensor or PIL image to numpy array.

			:param dtype: The optional dtype of the numpy array.
		"""
		super().__init__()
		self.dtype = dtype

	def forward(self, x: Union[list, np.ndarray, Tensor, Image.Image]) -> np.ndarray:
		return to_numpy(x, self.dtype)


class ToTensor(Module):
	def __init__(self, dtype: Optional[torch.dtype] = None, device: Optional[torch.device] = None):
		"""
			Convert a python list, numpy array or PIL image to pytorch tensor.
		"""
		super().__init__()
		self.dtype = dtype
		self.device = device

	def forward(self, x: Union[list, np.ndarray, Tensor, Image.Image]) -> Tensor:
		return to_tensor(x, self.dtype, self.device)


class ToList(Module):
	"""
		Convert a pytorch tensor, numpy array or PIL image to python list.
	"""
	def forward(self, x: Union[list, np.ndarray, Tensor, Image.Image]) -> list:
		return to_list(x)


class ToPIL(Module):
	def __init__(self, mode: Optional[str] = "RGB", permute_tensor: bool = True):
		"""
			Convert a pytorch tensor, numpy array or python list to PIL image.

			:param mode: Define the type and depth of a pixel in the image. (default: "RGB")
				See https://pillow.readthedocs.io/en/5.1.x/handbook/concepts.html#modes for details.
		"""
		super().__init__()
		self.mode = mode
		self.permute_tensor = permute_tensor

	def forward(self, x: Union[list, np.ndarray, Tensor, Image.Image]) -> Image.Image:
		return to_pil(x, self.mode, self.permute_tensor)


def to_numpy(
	x: Union[list, np.ndarray, Tensor, Image.Image],
	dtype: Optional[object] = None
) -> np.ndarray:
	"""
		:param x: The list, numpy array, pytorch tensor or pillow image to convert.
		:param dtype: The optional dtype of the numpy array.
	"""

	if isinstance(x, list) or isinstance(x, Image.Image):
		return np.asarray(x, dtype=dtype)
	elif isinstance(x, np.ndarray):
		return np.array(x, dtype=dtype)
	elif isinstance(x, Tensor):
		return np.array(x.cpu().numpy(), dtype=dtype)
	else:
		return np.asarray(x, dtype=dtype)


def to_tensor(
	x: Union[list, np.ndarray, Tensor, Image.Image],
	dtype: Optional[torch.dtype] = None,
	device: Optional[torch.device] = torch.device("cpu")
) -> Tensor:

	if isinstance(x, list):
		return torch.as_tensor(x, dtype=dtype, device=device)
	elif isinstance(x, np.ndarray):
		return torch.from_numpy(x.copy()).to(dtype).to(device)
	elif isinstance(x, Tensor):
		return x.to(dtype).to(device)
	elif isinstance(x, Image.Image):
		return to_tensor(to_numpy(x), dtype=dtype, device=device)
	else:
		return torch.as_tensor(x, dtype=dtype, device=device)


def to_list(
	x: Union[list, np.ndarray, Tensor, Image.Image]
) -> list:

	if isinstance(x, list):
		return x
	elif isinstance(x, np.ndarray) or isinstance(x, Tensor):
		return x.tolist()
	elif isinstance(x, Image.Image):
		return np.asarray(x).tolist()
	else:
		return list(x)


def to_pil(
	x: Union[list, np.ndarray, Tensor, Image.Image],
	mode: Optional[str] = "RGB",
	permute_tensor: bool = True,
) -> Image.Image:

	if isinstance(x, list):
		return Image.fromarray(np.asarray(x), mode)
	elif isinstance(x, np.ndarray):
		to_pil_image = tv.transforms.ToPILImage(mode)
		return to_pil_image(x)
	elif isinstance(x, Tensor):
		if permute_tensor:
			# Permute dimensions (height, width, channel) to (channel, height, width).
			x = x.permute(2, 0, 1)
		to_pil_image = tv.transforms.ToPILImage(mode)
		return to_pil_image(x)
	elif isinstance(x, Image.Image):
		return x
	else:
		return Image.fromarray(x, mode)
