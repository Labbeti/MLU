
from mlu.datasets.wrappers.base import DatasetWrapper
from torch.utils.data.dataset import Dataset


class CachedDataset(DatasetWrapper):
	def __init__(self, dataset: Dataset):
		super().__init__(dataset)
		self._cached_items = {}

	def __getitem__(self, *args, **kwargs):
		args_dict = locals()
		args_dict = {k: str(v) for k, v in args_dict.items()}

		if args_dict in self._cached_items.keys():
			item = self._cached_items[args_dict]
		else:
			item = super()._dataset[args_dict]
			self._cached_items[args_dict] = item
		return item
