
import csv
import os
import os.path as osp
import torchaudio

from enum import Enum
from py7zr import SevenZipFile

from torch import Tensor
from torch.nn import Module
from torch.utils.data.dataset import Dataset
from torchaudio.datasets.utils import download_url
from typing import Dict, List, Optional, Sized


class Subset(str, Enum):
	DEVELOPMENT: str = "development"
	EVALUATION: str = "evaluation"


FOLDER_NAME = "CLOTHO_V1"

FILES_INFOS = {
	"development": {
		"audio_archive": {
			"filename": "clotho_audio_development.7z",
			"url": "https://zenodo.org/record/3490684/files/clotho_audio_development.7z?download=1",
			"hash": "e3ce88561b317cc3825e8c861cae1ec6",
		},
		"captions": {
			"filename": "clotho_captions_development.csv",
			"url": "https://zenodo.org/record/3490684/files/clotho_captions_development.csv?download=1",
			"hash": "dd568352389f413d832add5cf604529f",
		},
		"metadata": {
			"filename": "clotho_metadata_development.csv",
			"url": "https://zenodo.org/record/3490684/files/clotho_metadata_development.csv?download=1",
			"hash": "582c18ee47cebdbe33dce1feeab53a56",
		},
	},
	"evaluation": {
		"audio_archive": {
			"filename": "clotho_audio_evaluation.7z",
			"url": "https://zenodo.org/record/3490684/files/clotho_audio_evaluation.7z?download=1",
			"hash": "4569624ccadf96223f19cb59fe4f849f",
		},
		"captions": {
			"filename": "clotho_captions_evaluation.csv",
			"url": "https://zenodo.org/record/3490684/files/clotho_captions_evaluation.csv?download=1",
			"hash": "1b16b9e57cf7bdb7f13a13802aeb57e2",
		},
		"metadata": {
			"filename": "clotho_metadata_evaluation.csv",
			"url": "https://zenodo.org/record/3490684/files/clotho_metadata_evaluation.csv?download=1",
			"hash": "13946f054d4e1bf48079813aac61bf77",
		},
	},
}


class ClothoV1(Dataset, Sized):
	"""
		Unofficial Clotho V1 pytorch dataset for DCASE2020 Task 6.

		Folder tree:

		root/
		└── CLOTHO_V1
			├── clotho_audio_development.7z
			├── clotho_audio_evaluation.7z
			├── clotho_captions_development.csv
			├── clotho_captions_evaluation.csv
			├── clotho_metadata_development.csv
			├── clotho_metadata_evaluation.csv
			├── development
			│         └── (2893 files, 5,4G)
			├── evaluation
			│         └── (1045 files, 2,0G)
			└── LICENSE
	"""

	def __init__(
		self,
		root: str,
		subset: str,
		download: bool = False,
		waveform_transform: Optional[Module] = None,
		captions_transform: Optional[Module] = None,
		waveform_cache: bool = False,
		verbose: int = 0,
	):
		"""
			:param root: The parent of the dataset root directory.
			:param subset: The subset of Clotho to use. Can be "development" or "evaluation".
			:param download: Download the dataset if download=True and if the dataset is not already downloaded.
				(default: False)
			:param waveform_transform: The transform to apply to waveforms (Tensor).
				(default: None)
			:param captions_transform: The transform to apply to captions with a list of 5 sentences (List[List[str]]).
				(default: None)
			:param waveform_cache: If True, store waveforms in memory otherwise load them from files.
				(default: False)
			:param verbose: Verbose level to use.
				(default: 0)
		"""
		assert subset in ["development", "evaluation"]

		super().__init__()
		self._dataset_root = osp.join(root, FOLDER_NAME)
		self._subset = subset
		self._download = download
		self._waveform_transform = waveform_transform
		self._captions_transform = captions_transform
		self._waveform_cache = waveform_cache
		self._verbose = verbose

		self._data_info = {}
		self._idx_to_filename = []
		self._waveforms = {}

		if self._download:
			self._download_dataset()

		self._prepare_data()

	def __getitem__(self, index: int) -> (Tensor, List[List[str]]):
		"""
			Get the audio data as 1D tensor and the matching captions as 5 sentences.

			:param index: The index of the item.
			:return: A tuple of audio data of shape (size,) and the 5 matching captions.
		"""
		waveform = self.get_waveform(index)
		captions = self.get_captions(index)
		return waveform, captions

	def __len__(self) -> int:
		"""
			:return: The number of items in the dataset.
		"""
		return len(self._data_info)

	def get_waveform(self, index: int) -> Tensor:
		"""
			:param index: The index of the item.
			:return: The audio data as 1D tensor.
		"""
		if not self._waveform_cache or index not in self._waveforms.keys():
			info = self._data_info[self.get_filename(index)]
			filepath = info["filepath"]
			waveform, _sample_rate = torchaudio.load(filepath)

			if self._waveform_cache:
				self._waveforms[index] = waveform

		else:
			waveform = self._waveforms[index]

		if self._waveform_transform is not None:
			waveform = self._waveform_transform(waveform)
		return waveform

	def get_captions(self, index: int) -> List[List[str]]:
		"""
			:param index: The index of the item.
			:return: The list of 5 captions of an item.
		"""
		info = self._data_info[self.get_filename(index)]
		captions = info["captions"]

		if self._captions_transform is not None:
			captions = self._captions_transform(captions)
		return captions

	def get_metadata(self, index: int) -> Dict[str, str]:
		"""
			Returns the metadata dictionary for a file.
			This dictionary contains :
				- "keywords": Contains the keywords of the item, separated by ";".
				- "sound_id": Id of the audio.
				- "sound_link": Link to the audio.
				- "start_end_samples": The range of the sound where it was extracted.
				- "manufacturer": The manufacturer of this file.
				- "licence": Link to the licence.

			:param index: The index of the item.
			:return: The metadata dictionary associated to the item.
		"""
		info = self._data_info[self.get_filename(index)]
		return info["metadata"]

	def get_filename(self, index: int) -> str:
		"""
			:param index: The index of the item.
			:return: The filename associated to the index.
		"""
		return self._idx_to_filename[index]

	def get_dataset_root(self) -> str:
		"""
			:return: The folder path of the dataset.
		"""
		return self._dataset_root

	def _download_dataset(self):
		if not osp.isdir(self._dataset_root):
			os.mkdir(self._dataset_root)

		if self._verbose >= 1:
			print("Download files for the dataset...")

		infos = FILES_INFOS[self._subset]

		# Download archives files
		for name, info in infos.items():
			filename, url, hash_ = info["filename"], info["url"], info["hash"]
			filepath = osp.join(self._dataset_root, filename)

			if not osp.isfile(filepath):
				if self._verbose >= 1:
					print(f"Download file '{filename}' from url '{url}'...")

				if osp.exists(filepath):
					raise RuntimeError(f"Object '{filepath}' already exists but it's not a file.")
				download_url(url, self._dataset_root, filename, hash_value=hash_, hash_type="md5")

		# Extract audio files from archives
		for name, info in infos.items():
			filename = info["filename"]
			filepath = osp.join(self._dataset_root, filename)
			extension = filename.split(".")[-1]

			if extension == "7z":
				extracted_path = osp.join(self._dataset_root, self._subset)

				if not osp.isdir(extracted_path):
					if self._verbose >= 1:
						print(f"Extract archive file '{filename}'...")

					archive_file = SevenZipFile(filepath)
					archive_file.extractall(self._dataset_root)
					archive_file.close()

	def _prepare_data(self):
		# Read filepath of .wav audio files
		dirpath_audio = osp.join(self._dataset_root, self._subset)
		self._data_info = {
			filename: {
				"filepath": osp.join(dirpath_audio, filename)
			}
			for filename in os.listdir(dirpath_audio)
		}

		files_infos = FILES_INFOS[self._subset]

		# Read captions info
		captions_filename = files_infos["captions"]["filename"]
		captions_filepath = osp.join(self._dataset_root, captions_filename)

		# Keys: file_name, caption_1, caption_2, caption_3, caption_4, caption_5
		with open(captions_filepath, "r") as file:
			reader = csv.DictReader(file)
			for row in reader:
				filename = row["file_name"]
				if filename in self._data_info.keys():
					self._data_info[filename]["captions"] = [
						row[caption_key] for caption_key in ["caption_1", "caption_2", "caption_3", "caption_4", "caption_5"]
					]
				else:
					raise RuntimeError(f"Found filename '{filename}' in CSV '{captions_filename}' but not the audio file.")

		# Read metadata info
		metadata_filename = files_infos["metadata"]["filename"]
		metadata_filepath = osp.join(self._dataset_root, metadata_filename)

		# Keys: file_name, keywords, sound_id, sound_link, start_end_samples, manufacturer, license
		with open(metadata_filepath, "r") as file:
			reader = csv.DictReader(file)
			for row in reader:
				filename = row["file_name"]
				row_copy = dict(row)
				row_copy.pop("file_name")

				if filename in self._data_info.keys():
					self._data_info[filename]["metadata"] = row_copy
				else:
					raise RuntimeError(f"Found filename '{filename}' in CSV '{metadata_filename}' but not the audio file.")

		self._idx_to_filename = [filename for filename in self._data_info.keys()]
