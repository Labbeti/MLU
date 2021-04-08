
from torch import Tensor
from typing import Optional


def is_binary(t: Tensor, dim: Optional[int] = None) -> Tensor:
	return t.eq(1.0).logical_or(t.eq(0.0)).all(dim)
