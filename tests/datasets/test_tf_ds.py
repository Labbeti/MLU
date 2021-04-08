
import timeit
from mlu.datasets.wrappers import TransformDataset
from torch.utils.data.dataset import Dataset


class DummyDataset(Dataset):
	def __init__(self, size: int = 100):
		super().__init__()
		self.size = size

	def __getitem__(self, idx: int) -> int:
		return idx

	def __len__(self) -> int:
		return self.size


def test_tf_ds():
	dataset = DummyDataset()
	transform = lambda x: x ** 2
	dataset = TransformDataset(dataset, transform=transform, index=None)

	s = 0
	for idx in range(len(dataset)):
		data = dataset[idx]
		s += data


def perf():
	print("Begin perf test")
	results = timeit.timeit(test_tf_ds, number=100000)
	print("End perf test")
	print(results)


if __name__ == "__main__":
	perf()
