
import inspect

from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Union, Iterator


class ObjectBuilder:
	def __init__(self, case_sensitive: bool = True, verbose: int = 0):
		super().__init__()
		self._case_sensitive = case_sensitive
		self._verbose = verbose

		self._alias_to_id = {}
		self._classes_or_funcs = {}
		self._default_kwargs = {}

	def register_class_or_func(
		self,
		class_or_func: object,
		aliases: Optional[Union[str, Iterable[str]]] = None,
		default_kwargs: Optional[Dict[str, Any]] = None,
	):
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
			raise RuntimeError("Alias must be None, str or a Iterable[str].")

		id_ = class_or_func.__name__
		id_ = self._process(id_)
		aliases.add(id_)

		if id_ in self._alias_to_id.keys():
			if self._verbose >= 1:
				print(f"WARNING: Overwrite class or func with the same alias '{id_}'.")
			self.pop(id_)

		for alias in aliases:
			alias = self._process(alias)
			self._alias_to_id[alias] = id_

		self._classes_or_funcs[id_] = class_or_func
		self._default_kwargs[id_] = default_kwargs

	def register(self, obj: Any, predicate_class_or_func: Optional[Callable[[Any], bool]] = None):
		if inspect.isclass(obj) or inspect.isfunction(obj):
			# Register class or function
			if self._verbose >= 1:
				print(f"Register class or function '{obj.__name__}'.")
			if predicate_class_or_func is None or predicate_class_or_func(obj):
				self.register_class_or_func(obj)

		elif inspect.ismodule(obj):
			# Get submodules, classes and functions defined in the module "obj"
			if self._verbose >= 1:
				print(f"Register module '{obj.__name__}'.")
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
				print(f"Register iterable '{type(obj)}'.")
			for member in obj:
				self.register(member, predicate_class_or_func)

	def build(self, name: str, *args, **kwargs) -> object:
		name = self._process(name)
		class_or_func = self._get_class_or_func(name)
		# func.__code__.co_varnames => Tuple[str, ...]
		# func.__defaults__ => Tuple[Any, ...]
		default_kwargs = dict(self._get_default_kwargs(name))
		default_kwargs.update(kwargs)
		return class_or_func(*args, **default_kwargs)

	def add_alias(self, name: str, alias: str):
		name = self._process(name)
		alias = self._process(alias)

		id_ = self._alias_to_id[name]
		self._alias_to_id[alias] = id_

	def add_aliases(self, name: str, aliases: Iterable[str]):
		for alias in aliases:
			self.add_alias(name, alias)

	def pop(self, alias: str) -> (Callable, Dict[str, Any], List[str]):
		alias = self._process(alias)
		if alias not in self._alias_to_id.keys():
			raise RuntimeError(f"Unknown object '{alias}'.")

		old_id = self._alias_to_id[alias]
		old_aliases = self._get_aliases(old_id)
		for old_alias in old_aliases:
			self._alias_to_id.pop(old_alias)

		class_or_func = self._classes_or_funcs.pop(old_id)
		default_kwargs = self._default_kwargs.pop(old_id)
		return class_or_func, default_kwargs, old_aliases

	def clear(self):
		self._alias_to_id.clear()
		self._classes_or_funcs.clear()
		self._default_kwargs.clear()

	def keys(self) -> Iterator:
		for key in self._classes_or_funcs.keys():
			yield key

	def values(self) -> Iterator:
		for value in self._classes_or_funcs.values():
			yield value

	def items(self) -> Iterator:
		for key, value in self._classes_or_funcs.items():
			yield key, value

	def _get_class_or_func(self, alias: str) -> Callable:
		alias = self._process(alias)
		if alias not in self._alias_to_id.keys():
			raise RuntimeError(f"Unknown object '{alias}'.")
		id_ = self._alias_to_id[alias]
		class_or_func = self._classes_or_funcs[id_]
		return class_or_func

	def _get_default_kwargs(self, alias: str) -> Dict[str, Any]:
		alias = self._process(alias)
		if alias not in self._alias_to_id.keys():
			raise RuntimeError(f"Unknown object '{alias}'.")
		id_ = self._alias_to_id[alias]
		default_kwargs = self._default_kwargs[id_]
		return default_kwargs

	def _get_aliases(self, alias: str) -> Set[str]:
		alias = self._process(alias)
		if alias not in self._alias_to_id.keys():
			raise RuntimeError(f"Unknown object '{alias}'.")
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
	from abc import ABC
	from mlu.metrics import classification, text
	from mlu.metrics.base import Metric

	builder = ObjectBuilder()
	predicate = lambda obj: inspect.isclass(obj) and issubclass(obj, Metric) and not isinstance(obj, ABC)
	builder.register([classification, text], predicate)
	return builder


def get_transform_builder() -> ObjectBuilder:
	from abc import ABC
	from mlu.transforms import image, spectrogram, waveform
	from mlu.transforms.base import Transform

	builder = ObjectBuilder()
	predicate = lambda obj: inspect.isclass(obj) and issubclass(obj, Transform) and not isinstance(obj, ABC)
	builder.register([image, spectrogram, waveform], predicate)
	return builder
