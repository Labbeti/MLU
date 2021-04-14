
from typing import Iterable, Iterator, Optional, Sized, Protocol


class SizedIterable(Iterable, Sized, Protocol):
	pass


class ZipCycle(Iterable, Sized):
	def __init__(self, *iterables: SizedIterable, mode: str = 'max'):
		"""
			Zip through a list of iterables and sized objects of different lengths.
			Reset the iterators when there and finish loop when the longest one is finished.

			Example :

			>>> r1 = range(1, 4)
			>>> r2 = range(1, 6)
			>>> cycle = ZipCycle(r1, r2)
			>>> for v1, v2 in cycle:
			>>> 	print('(', v1, ',', v2, ')')
			... ( 1 , 1 )
			... ( 2 , 2 )
			... ( 3 , 3 )
			... ( 1 , 4 )
			... ( 2 , 5 )

			:param iterables: A list of Sized Iterables to browse. Can not be an empty list.
			:param mode: The mode to use during iteration. (default: 'max')
				If mode = 'min', the iterator will stop when the shortest iterable is finished. (like in the built-in 'zip' python)
				If mode = 'max', the iterator will stop when the longest iterable is finished. (like in the example above)
				If mode = 'inf', the iterator will never stop and cycle indefinitely on iterables stored.
		"""
		assert mode in ['min', 'max', 'inf'], f'Available modes are "{("min", "max", "inf")}".'
		assert len(iterables) > 0

		lens = [len(iterable) for iterable in iterables]
		for len_ in lens:
			if len_ == 0:
				raise RuntimeError('An iterable stored in ZipCycle is empty.')

		self._iterables = iterables
		self._mode = mode
		self._counter = 0

	def __iter__(self) -> Iterator[list]:
		cur_iters = [iter(iterable) for iterable in self._iterables]
		cur_count = [0 for _ in self._iterables]

		self._counter = 0
		while len(self) is None or self._counter < len(self):
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
			self._counter += 1

	def __len__(self) -> Optional[int]:
		if len(self._iterables) == 0:
			return 0
		elif self._mode == 'min':
			return min(len(iterable) for iterable in self._iterables)
		elif self._mode == 'max':
			return max(len(iterable) for iterable in self._iterables)
		else:  # mode == 'inf'
			return None

	def get_counter(self) -> int:
		return self._counter

	def get_mode(self) -> str:
		return self._mode

	def set_mode(self, mode: str):
		assert mode in ['min', 'max', 'inf'], f'Mode must be one of {("min", "max", "inf")}.'
		self._mode = mode
