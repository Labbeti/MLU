
from mlu.utils.typing.classes import SizedIterable
from typing import Iterable, Optional, Sized


class ZipCycle(Iterable, Sized):
	def __init__(self, *iterables: SizedIterable, policy: str = "max"):
		"""
			Zip through a list of iterables and sized objects of different lengths.
			Reset the iterators when there and finish iteration when the longest one is over.

			Example :

			>>> r1 = range(1, 4)
			>>>	r2 = range(1, 6)
			>>> cycle = ZipCycle([r1, r2])
			>>> for v1, v2 in cycle:
			>>> 	print("(", v1, ",", v2, ")")
			...	( 1 , 1 )
			...	( 2 , 2 )
			...	( 3 , 3 )
			...	( 1 , 4 )
			...	( 2 , 5 )

			:param iterables: A list of Sized Iterables to browse. Must not be an empty list.
			:param policy: The policy to use during iteration. (default: "max")
				If policy = "min", the iterator will stop when the first iterable is finished. (like in the built-in "zip" python)
				If policy = "max", the iterator will stop when the last iterable is finished. (like in the example above)
				If policy = "inf", the iterator will never stop and cycle indefinitely on iterables stored.
		"""
		assert policy in ["min", "max", "inf"], f"Available policies are '{('min', 'max', 'inf')}'."
		assert len(iterables) > 0

		lens = [len(iterable) for iterable in iterables]
		for len_ in lens:
			if len_ == 0:
				raise RuntimeError("An iterable is empty.")

		self._iterables = iterables
		self._policy = policy

	def __iter__(self) -> list:
		cur_iters = [iter(iterable) for iterable in self._iterables]
		cur_count = [0 for _ in self._iterables]

		i = 0
		while len(self) is None or i < len(self):
			items = []

			for i, _ in enumerate(cur_iters):
				if cur_count[i] < len(self._iterables[i]):
					item = next(cur_iters[i])
					cur_count[i] += 1
				else:
					cur_iters[i] = iter(self._iterables[i])
					item = next(cur_iters[i])
					cur_count[i] = 1
				items.append(item)

			yield items
			i += 1

	def __len__(self) -> Optional[int]:
		if self._policy == "min":
			return min(len(iterable) for iterable in self._iterables)
		elif self._policy == "max":
			return max(len(iterable) for iterable in self._iterables)
		else:  # policy == "inf"
			return None

	def set_policy(self, policy: str):
		assert policy in ["min", "max", "inf"]
		self._policy = policy
