
import csv
import os
import os.path as osp
import requests
import torch
import torchaudio

from enum import Enum, IntEnum
from torch import Tensor
from torch.nn import Module
from torch.utils.data.dataset import Dataset
from typing import Optional, Sized, Union


class AudioSetSubset(str, Enum):
	BALANCED: str = "balanced"
	UNBALANCED: str = "unbalanced"
	EVAL: str = "eval"


CLASS_LABELS_INDICES_URL = "http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/class_labels_indices.csv"


class AudioSet(Dataset, Sized):
	"""
		Unofficial AudioSet pytorch dataset.

		Dataset files and folders tree :

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
		AudioSetSubset.BALANCED: osp.join("data", "balanced_train_segments.csv"),
		AudioSetSubset.UNBALANCED: osp.join("data", "unbalanced_train_segments.csv"),
		AudioSetSubset.EVAL: osp.join("data", "eval_segments.csv"),
	}

	SUBSET_DIRPATH = {
		AudioSetSubset.BALANCED: osp.join("data", "data", "balanced_train_segments", "audio"),
		AudioSetSubset.UNBALANCED: osp.join("data", "data", "unbalanced_train_segments", "audio"),
		AudioSetSubset.EVAL: osp.join("data", "data", "eval_segments", "audio"),
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
		subset: Union[str, AudioSetSubset],
		transform: Optional[Module] = None,
		target_transform: Optional[Module] = None,
		verbose: int = 0
	):
		"""
			Unofficial AudioSet pytorch dataset.

			Example :

			>>> from mlu.datasets import AudioSet
			>>> from mlu.nn import MultiHot
			>>> dataset = AudioSet("../data", "balanced", target_transform=MultiHot(527))

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
			subsets_names = [s.value for s in list(AudioSetSubset)]
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

	def __getitem__(self, index: int) -> (Tensor, Tensor):
		"""
			Returns the audio data with labels.

			:param index: The index of the audio sample.
			:return: (Audio data as tensor, labels classes indexes as tensor of indexes of classes)
		"""
		audio_data = self.get_audio(index)
		target = self.get_target(index)

		return audio_data, target

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

	def get_audio(self, index: int) -> Tensor:
		"""
			:param index: The index of the data.
			:return: The raw waveform from the audio file as Tensor.
		"""
		filepath = self.get_filepath(index)
		audio_data, _sample_rate = torchaudio.load(filepath)

		if self._transform is not None:
			audio_data = self._transform(audio_data)

		return audio_data

	def get_target(self, index: int) -> Tensor:
		"""
			:param index: The index of the data.
			:return: The label classes indexes for a specific sample index.
		"""
		target = self._metadata[self._subset][index]["labels"]

		if self._target_transform is not None:
			target = self._target_transform(target)

		return target

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

	def _load_labels_table(self):
		def rec_basename(path: str, n: int) -> str:
			for _ in range(n):
				path = osp.basename(path)
			return path

		abs_current_fpath = osp.join(os.getcwd(), __file__)
		class_labels_indices_fpath = osp.join(rec_basename(abs_current_fpath, 3), "class_labels_indices.csv")
		if not osp.isfile(class_labels_indices_fpath):
			with open(class_labels_indices_fpath, "wb") as file:
				req = requests.get(CLASS_LABELS_INDICES_URL)
				file.write(req.content)

		table_filepath = class_labels_indices_fpath

		if not osp.isfile(table_filepath):
			raise RuntimeError(f"Invalid CSV class labels indices file '{table_filepath}'.")

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
			print(
				f"Info: Found {num_entries} metadata entries and {len(subset_metadata)} audio files matching with entries. "
				f"(missing {num_entries - len(subset_metadata)} files)"
			)
