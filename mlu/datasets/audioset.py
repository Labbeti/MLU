
import csv
import os.path as osp
import torch
import torchaudio

from enum import Enum, IntEnum
from torch import Tensor, IntTensor
from torch.nn import Module
from torch.utils.data.dataset import Dataset
from typing import Optional, Sized, Union


class Subset(str, Enum):
	BALANCED: str = "balanced"
	UNBALANCED: str = "unbalanced"
	EVAL: str = "eval"


class AudioSet(Dataset, Sized):
	"""
		Unofficial AudioSet pytorch dataset.

		root/
		└── data/
			├── balanced_train_segments.csv
			├── unbalanced_train_segments.csv
			├── eval_segments.csv
			└── data/
				├── balanced_train_segments/
				│   └── audio/
				│   	└── (22160 files, ~19GB)
				├── unbalanced_train_segments/
				│   └── audio/
				│ 		└── (2041789 files)
				└── eval_segments/
					└── audio/
						└── (20371 files, ~18GB)
	"""

	METADATA_FILEPATH = {
		Subset.BALANCED: osp.join("data", "balanced_train_segments.csv"),
		Subset.UNBALANCED: osp.join("data", "unbalanced_train_segments.csv"),
		Subset.EVAL: osp.join("data", "eval_segments.csv"),
	}

	SUBSET_DIRPATH = {
		Subset.BALANCED: osp.join("data", "data", "balanced_train_segments", "audio"),
		Subset.UNBALANCED: osp.join("data", "data", "unbalanced_train_segments", "audio"),
		Subset.EVAL: osp.join("data", "data", "eval_segments", "audio"),
	}

	AUDIO_FILE_EXTENSION = "flac"

	# Column indexes in segments metadata files :
	# "balanced_train_segments.csv", "unbalanced_train_segments.csv", "eval_segments.csv"
	class MetaIdx(IntEnum):
		YTID = 0
		START_SECONDS = 1
		END_SECONDS = 2
		POSITIVE_LABELS = 3

	# Column indexes in "labels.csv" metadata file.
	class LabelsIdx(IntEnum):
		INDEX = 0
		MID = 1
		DISPLAY_NAME = 2

	def __init__(
		self,
		root: str,
		subset: Union[str, Subset],
		transform: Optional[Module] = None,
		target_transform: Optional[Module] = None,
		verbose: int = 0
	):
		"""
			Constructor of the AudioSet dataset.

			:param root: Directory path to the dataset root architecture.
			:param subset: The name of the subset to load. Must be "balanced", "unbalanced" or "eval".
			:param transform: The transform to apply to the raw audio data extracted.
				The default shape of raw audio is (1, 320000).
				(default: None)
			:param target_transform: The transform to apply to the tensor of indexes.
				The default shape of labels is (nb_labels, ), where 'nb_labels' depends of the number of labels present in
				the corresponding audio sample.
				(default: None)
			:param verbose: The verbose level.
				0 print only fatal errors,
				1 for print errors and warnings,
				2 for print errors, warning and info.
				(default: 0)
		"""

		if isinstance(subset, str):
			subset = subset.lower()
			subsets_names = [s.value for s in list(Subset)]
			if subset not in subsets_names:
				raise RuntimeError(f"Invalid subset name '{subset}'. Must be one of : {str(subsets_names)}")
			subset = Subset(subset)

		self._dataset_root = root
		self._subset = subset
		self._transform = transform
		self._target_transform = target_transform
		self._verbose = verbose

		self._convert_table = {}  # Dict[str, Dict[str, Union[int, str]]]
		self._metadata = {}  # Dict[Subset, List[Dict[str, Union[str, List[int]]]]

		self._check_arguments()
		self._load_labels_table()
		self._load_subset_metadata()

	def __getitem__(self, index: int) -> (Tensor, IntTensor):
		"""
			Returns the audio data with labels.

			:param index: The index of the audio sample.
			:return: (Audio data as tensor, labels classes indexes as tensor of indexes of classes)
		"""
		audio_data = self.get_raw_data(index)
		labels = self.get_raw_labels(index)

		if self._transform is not None:
			audio_data = self._transform(audio_data)

		if self._target_transform is not None:
			labels = self._target_transform(labels)

		return audio_data, labels

	def __len__(self) -> int:
		"""
			:return: The number of examples in the current subset.
		"""
		return len(self._metadata[self._subset])

	def get_filepath(self, index: int) -> str:
		"""
			:param index: The index of the audio.
			:return: The audio filepath for a specific index.
		"""
		return self._metadata[self._subset][index]["filepath"]

	def get_raw_data(self, index: int) -> Tensor:
		"""
			:param index: The index of the audio.
			:return: The raw waveform from the audio file as Tensor.
		"""
		filepath = self.get_filepath(index)
		audio_data, _sample_rate = torchaudio.load(filepath)
		return audio_data

	def get_raw_labels(self, index: int) -> IntTensor:
		"""
			:param index: The index of the classes.
			:return: The label classes indexes for a specific sample index.
		"""
		return self._metadata[self._subset][index]["labels"]

	def get_num_classes(self) -> int:
		"""
			:return: The number of classes present in metadata 'labels.csv' file.
		"""
		return len(self._convert_table)

	def _check_arguments(self):
		dataset_dirpath = self._dataset_root
		if not osp.isdir(dataset_dirpath):
			raise RuntimeError(f"Cannot find dataset root directory path '{dataset_dirpath}'.")

		subset_dirpath = osp.join(self._dataset_root, self.SUBSET_DIRPATH[self._subset])
		if not osp.isdir(subset_dirpath):
			raise RuntimeError(f"Cannot find waveform directory '{subset_dirpath}'.")

		meta_filepath = osp.join(self._dataset_root, self.METADATA_FILEPATH[self._subset])
		if not osp.isfile(meta_filepath):
			raise RuntimeError(f"Cannot find CSV metadata file '{meta_filepath}'.")

		metadata_dirpath = osp.join(osp.dirname(osp.dirname(__file__)), "metadata")
		table_filepath = osp.join(metadata_dirpath, "labels.csv")
		if not osp.isfile(table_filepath):
			raise RuntimeError(f"Cannot find CSV labels table file '{table_filepath}'.")

	def _load_labels_table(self):
		metadata_dirpath = osp.join(osp.dirname(osp.dirname(__file__)), "metadata")
		table_filepath = osp.join(metadata_dirpath, "labels.csv")

		if not osp.isfile(table_filepath):
			raise RuntimeError(f"Cannot find CSV labels table file '{table_filepath}', it is not a file.")

		with open(table_filepath, "r") as table_file:
			reader = csv.reader(table_file, skipinitialspace=True, strict=True)

			for _ in range(2):
				next(reader)

			convert_table = {}

			for info in reader:
				index = int(info[self.LabelsIdx.INDEX])
				mid = info[self.LabelsIdx.MID]
				display_name = info[self.LabelsIdx.DISPLAY_NAME]

				convert_table[mid] = {
					"index": index,
					"display_name": display_name,
				}

		self._convert_table = convert_table

	def _load_subset_metadata(self):
		meta_filepath = osp.join(self._dataset_root, self.METADATA_FILEPATH[self._subset])

		if not osp.isfile(meta_filepath):
			raise RuntimeError(f"Cannot find CSV metadata file '{meta_filepath}'.")

		subset_metadata = []

		with open(meta_filepath, "r") as meta_file:
			reader = csv.reader(meta_file, skipinitialspace=True, strict=True)

			# Skip the comments
			for _ in range(3):
				next(reader)

			num_entries = 0
			# Read metadata
			for i, metadata in enumerate(reader):
				ytid = metadata[self.MetaIdx.YTID]
				start_ms = int(float(metadata[self.MetaIdx.START_SECONDS]) * 1000.0)
				end_ms = int(float(metadata[self.MetaIdx.END_SECONDS]) * 1000.0)

				audio_filename = "{:s}_{:d}_{:d}.{:s}".format(ytid, start_ms, end_ms, self.AUDIO_FILE_EXTENSION)
				audio_filepath = osp.join(self._dataset_root, self.SUBSET_DIRPATH[self._subset], audio_filename)

				if osp.isfile(audio_filepath):
					if self._verbose >= 1:
						print(f"Warning: Found a metadata entry for '{ytid}' but not the audio FLAC file in '{audio_filepath}'.")

					labels = metadata[self.MetaIdx.POSITIVE_LABELS].split(",")
					for label in labels:
						if label not in self._convert_table.keys():
							raise RuntimeError(f"Found unknown label '{label}'.")
					labels = [self._convert_table[label]["index"] for label in labels]
					labels = torch.as_tensor(labels, dtype=torch.int)

					subset_metadata.append({
						"filepath": audio_filepath,
						"labels": labels,
					})

				num_entries += 1

		self._metadata[self._subset] = subset_metadata
		if self._verbose >= 2:
			print(f"Info: Found {num_entries} metadata entries and {len(subset_metadata)} audio files matching with entries.")
