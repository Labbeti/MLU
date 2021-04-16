
import inspect

from abc import ABC
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Set, Tuple, Union


class ObjectBuilder:
	def __init__(self, case_sensitive: bool = True, verbose: int = 0):
		"""
			ObjectBuilder for build objects by name.

			TODO : add submodules of a module.

			:param case_sensitive: If True, the names registered will be in convert to lower case. (default: True)
			:param verbose: Verbose level of the class. Use 1 for print information and warnings. (default: 0)
		"""
		super().__init__()
		self._case_sensitive = case_sensitive
		self._verbose = verbose

		self._alias_to_id = {}
		self._classes_or_funcs = {}
		self._default_kwargs = {}

	def register(self, obj: Any, predicate_class_or_func: Optional[Callable[[Any], bool]] = None):
		"""
			Register a class, function, elements of a module or elements of an iterable recursively.

			:param obj: The object to register.
			:param predicate_class_or_func: The predicate to use for classes or functions. (default: None)
		"""
		if inspect.isclass(obj) or inspect.isfunction(obj):
			# Register class or function
			if self._verbose >= 1:
				print(f'Register class or function "{obj.__name__}".')
			self.register_class_or_func(obj, predicate_class_or_func=predicate_class_or_func)

		elif inspect.ismodule(obj):
			# Get submodules, classes and functions defined in the module 'obj'
			if self._verbose >= 1:
				print(f'Register module "{obj.__name__}".')
			predicate = (
				lambda member: (
					((inspect.isclass(member) or inspect.isfunction(member)) and member.__module__ == obj.__name__) or
					(inspect.ismodule(member) and member.__name__.startswith(obj.__name__))
				)
			)
			members = inspect.getmembers(obj, predicate)
			members = [member for _name, member in members]
			self.register(members, predicate_class_or_func)

		elif isinstance(obj, Iterable):
			# Recursive call on each element of the iterable
			if self._verbose >= 1:
				print(f'Register iterable "{type(obj)}".')
			for member in obj:
				self.register(member, predicate_class_or_func)

	def register_class_or_func(
		self,
		class_or_func: object,
		aliases: Optional[Union[str, Iterable[str]]] = None,
		default_kwargs: Optional[Dict[str, Any]] = None,
		predicate_class_or_func: Optional[Callable[[Any], bool]] = None,
	):
		"""
			Register a class or function.

			:param class_or_func: The object to register.
			:param aliases: The optional aliases for the object. (default: None)
			:param default_kwargs: The default kwargs for building the object. (default: None)
			:param predicate_class_or_func: The predicate for the object to register.
				If predicate returns False, the object is not registered.
				(default: None)
		"""
		if predicate_class_or_func is not None and not predicate_class_or_func(class_or_func):
			return
		assert inspect.isclass(class_or_func) or inspect.isfunction(class_or_func)
		if default_kwargs is None:
			default_kwargs = {}

		if aliases is None:
			aliases = set()
		elif isinstance(aliases, str):
			aliases = {aliases}
		elif isinstance(aliases, Iterable):
			aliases = set(aliases)
		else:
			raise RuntimeError('Alias must be None, str or a Iterable[str].')

		id_ = class_or_func.__name__
		id_ = self._process(id_)
		aliases.add(id_)

		if id_ in self._alias_to_id.keys():
			if self._verbose >= 1:
				print(f'WARNING: Overwrite class or func with the same alias "{id_}".')
			self.pop(id_)

		for alias in aliases:
			alias = self._process(alias)
			self._alias_to_id[alias] = id_

		self._classes_or_funcs[id_] = class_or_func
		self._default_kwargs[id_] = default_kwargs

	def build(self, name: str, filter_kwargs: bool = True, *args, **kwargs) -> object:
		"""
			Build the object with a specific name.

			:param name: The name of the object.
			:param filter_kwargs: If True, kwargs keys will be filtered with parameters arguments for build the object.
			:param args: The positional arguments for build the object.
			:param kwargs: The named arguments for build the object.
		"""
		# func.__code__.co_varnames => Tuple[str, ...]
		# func.__defaults__ => Tuple[Any, ...]
		name = self._process(name)
		class_or_func = self._get_class_or_func(name)

		if filter_kwargs:
			if inspect.isfunction(class_or_func) and hasattr(class_or_func, '__code__'):
				parameters_names = class_or_func.__code__.co_varnames
			elif inspect.isclass(class_or_func) and hasattr(class_or_func, '__init__'):
				parameters_names = class_or_func.__init__.__code__.co_varnames
			else:
				raise RuntimeError(f'Invalid class or func "{class_or_func.__name__}".')

			kwargs = {k: v for k, v in kwargs.items() if k in parameters_names}

		default_kwargs = dict(self._get_default_kwargs(name))
		default_kwargs.update(kwargs)
		return class_or_func(*args, **default_kwargs)

	def add_alias(self, old_alias: str, new_alias: str):
		"""
			Add alias for building an object.

			:param old_alias: The original name of the object. Can be an alias already stored.
			:param new_alias: The alias to add.
		"""
		old_alias = self._process(old_alias)
		new_alias = self._process(new_alias)

		if old_alias not in self._alias_to_id.keys():
			raise RuntimeError(f'Unknown object "{old_alias}".')
		id_ = self._alias_to_id[old_alias]
		self._alias_to_id[new_alias] = id_

	def add_aliases(self, name: str, aliases: Iterable[str]):
		"""
			Add aliases for building an object.

			:param name: The original name of the object. Can be an alias already stored.
			:param aliases: The aliases to add.
		"""
		for alias in aliases:
			self.add_alias(name, alias)

	def pop(self, alias: str) -> (Callable, Dict[str, Any], List[str]):
		"""
			Remove an object with a name or alias.

			:param alias: The alias of the object to remove.
		"""
		alias = self._process(alias)
		if alias not in self._alias_to_id.keys():
			raise RuntimeError(f'Unknown object "{alias}".')

		old_id = self._alias_to_id[alias]
		old_aliases = self._get_aliases(old_id)
		for old_alias in old_aliases:
			self._alias_to_id.pop(old_alias)

		class_or_func = self._classes_or_funcs.pop(old_id)
		default_kwargs = self._default_kwargs.pop(old_id)
		return class_or_func, default_kwargs, old_aliases

	def clear(self):
		"""
			Clear all objects and aliases registered.
		"""
		self._alias_to_id.clear()
		self._classes_or_funcs.clear()
		self._default_kwargs.clear()

	def keys(self) -> Iterator[str]:
		"""
			Returns an Iterator on object names.
		"""
		for key in self._classes_or_funcs.keys():
			yield key

	def values(self) -> Iterator[Callable]:
		"""
			Returns an Iterator on classes and functions types.
		"""
		for value in self._classes_or_funcs.values():
			yield value

	def items(self) -> Iterator[Tuple[str, Callable]]:
		"""
			Returns an Iterator on names and classes and functions types.
		"""
		for key, value in self._classes_or_funcs.items():
			yield key, value

	def _get_class_or_func(self, alias: str) -> Callable:
		alias = self._process(alias)
		if alias not in self._alias_to_id.keys():
			raise RuntimeError(f'Unknown object "{alias}".')
		id_ = self._alias_to_id[alias]
		class_or_func = self._classes_or_funcs[id_]
		return class_or_func

	def _get_default_kwargs(self, alias: str) -> Dict[str, Any]:
		alias = self._process(alias)
		if alias not in self._alias_to_id.keys():
			raise RuntimeError(f'Unknown object "{alias}".')
		id_ = self._alias_to_id[alias]
		default_kwargs = self._default_kwargs[id_]
		return default_kwargs

	def _get_aliases(self, alias: str) -> Set[str]:
		alias = self._process(alias)
		if alias not in self._alias_to_id.keys():
			raise RuntimeError(f'Unknown object "{alias}".')
		main_id = self._alias_to_id[alias]
		aliases = set()
		for alias, id_ in self._alias_to_id.items():
			if main_id == id_:
				aliases.add(alias)
		return aliases

	def _process(self, name):
		if not self._case_sensitive:
			name = name.lower()
		return name


def get_metric_builder() -> ObjectBuilder:
	"""
		Returns a ObjectBuilder with MLU metrics registered.
	"""
	from mlu.metrics import classification, text
	from mlu.metrics.base import Metric

	modules = [classification, text]
	predicate = lambda obj: inspect.isclass(obj) and issubclass(obj, Metric) and not isinstance(obj, ABC)
	builder = ObjectBuilder()
	builder.register(modules, predicate)

	return builder


def get_transform_builder() -> ObjectBuilder:
	"""
		Returns a ObjectBuilder with MLU transforms registered.
	"""
	from mlu.transforms import image, spectrogram, waveform
	from mlu.transforms.base import Transform

	modules = [image, spectrogram, waveform]
	predicate = lambda obj: inspect.isclass(obj) and issubclass(obj, Transform) and not isinstance(obj, ABC)
	builder = ObjectBuilder()
	builder.register(modules, predicate)

	return builder
