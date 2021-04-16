
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
from typing import Dict, List, Optional


class ClothoV1Subset(str, Enum):
	DEVELOPMENT: str = 'development'
	EVALUATION: str = 'evaluation'
	TEST: str = 'test'


FOLDER_NAME: str = 'CLOTHO_V1'
SAMPLE_RATE: int = 44100
AUDIO_MAX_LENGTH: int = 30  # in seconds

FILES_INFOS = {
	'development': {
		'audio_archive': {
			'filename': 'clotho_audio_development.7z',
			'url': 'https://zenodo.org/record/3490684/files/clotho_audio_development.7z?download=1',
			'hash': 'e3ce88561b317cc3825e8c861cae1ec6',
		},
		'captions': {
			'filename': 'clotho_captions_development.csv',
			'url': 'https://zenodo.org/record/3490684/files/clotho_captions_development.csv?download=1',
			'hash': 'dd568352389f413d832add5cf604529f',
		},
		'metadata': {
			'filename': 'clotho_metadata_development.csv',
			'url': 'https://zenodo.org/record/3490684/files/clotho_metadata_development.csv?download=1',
			'hash': '582c18ee47cebdbe33dce1feeab53a56',
		},
	},
	'evaluation': {
		'audio_archive': {
			'filename': 'clotho_audio_evaluation.7z',
			'url': 'https://zenodo.org/record/3490684/files/clotho_audio_evaluation.7z?download=1',
			'hash': '4569624ccadf96223f19cb59fe4f849f',
		},
		'captions': {
			'filename': 'clotho_captions_evaluation.csv',
			'url': 'https://zenodo.org/record/3490684/files/clotho_captions_evaluation.csv?download=1',
			'hash': '1b16b9e57cf7bdb7f13a13802aeb57e2',
		},
		'metadata': {
			'filename': 'clotho_metadata_evaluation.csv',
			'url': 'https://zenodo.org/record/3490684/files/clotho_metadata_evaluation.csv?download=1',
			'hash': '13946f054d4e1bf48079813aac61bf77',
		},
	},
	'test': {
		'audio_archive': {
			'filename': 'clotho_audio_test.7z',
			'url': "",
			'hash': '9b3fe72560a621641ff4351ba1154349',
		},
		'metadata': {
			'filename': 'clotho_metadata_test.csv',
			'url': "",
			'hash': '52f8ad01c229a310a0ff8043df480e21',
		},
	}
}


class ClothoV1(Dataset):
	"""
		Unofficial Clotho V1 pytorch dataset for DCASE 2020 Task 6.
		Audio are waveform sounds of 15 to 30 seconds, sampled at 44100 Hz.
		Targets are a list of 5 different sentences describing each audio sample.

		Clotho V1 Paper : https://arxiv.org/pdf/1910.09387.pdf

		Dataset folder tree:

		root/
		└── CLOTHO_V1
			├── clotho_audio_development.7z
			├── clotho_audio_evaluation.7z
			├── clotho_captions_development.csv
			├── clotho_captions_evaluation.csv
			├── clotho_metadata_development.csv
			├── clotho_metadata_evaluation.csv
			├── development
			│         └── (2893 files, ~5.4G)
			├── evaluation
			│         └── (1045 files, ~2.0G)
			└── LICENSE
	"""

	def __init__(
		self,
		root: str,
		subset: str,
		download: bool = False,
		audio_transform: Optional[Module] = None,
		captions_transform: Optional[Module] = None,
		audio_cache: bool = False,
		verbose: int = 0,
	):
		"""
			:param root: The parent of the dataset root directory. The data will be stored in the 'CLOTHO_V1' subdirectory.
			:param subset: The subset of Clotho to use. Can be 'development', 'evaluation' or 'test'.
			:param download: Download the dataset if download=True and if the dataset is not already downloaded.
				(default: False)
			:param audio_transform: The transform to apply to waveforms (Tensor).
				(default: None)
			:param captions_transform: The transform to apply to captions with a list of 5 sentences (List[str]).
				(default: None)
			:param audio_cache: If True, store audio waveforms into RAM memory after loading them from files.
				Can increase the data loading process time performance but requires enough RAM to store the data.
				(default: False)
			:param verbose: Verbose level to use. Can be 0 or 1.
				(default: 0)
		"""
		assert subset in ['development', 'evaluation', 'test']

		super().__init__()
		self._dataset_root = osp.join(root, FOLDER_NAME)
		self._subset = subset
		self._download = download
		self._audio_transform = audio_transform
		self._captions_transform = captions_transform
		self._audio_cache = audio_cache
		self._verbose = verbose

		self._data_info = {}
		self._idx_to_filename = []
		self._waveforms = {}

		if self._download:
			self._download_dataset()

		self._prepare_data()

	def __getitem__(self, index: int) -> (Tensor, List[str]):
		"""
			Get the audio data as 1D tensor and the matching captions as 5 sentences.

			:param index: The index of the item.
			:return: A tuple of audio data of shape (size,) and the 5 matching captions.
		"""
		waveform = self.get_audio(index)
		captions = self.get_captions(index)
		return waveform, captions

	def __len__(self) -> int:
		"""
			:return: The number of items in the dataset.
		"""
		return len(self._data_info)

	def get_audio(self, index: int) -> Tensor:
		"""
			:param index: The index of the item.
			:return: The audio data as 1D tensor.
		"""
		if not self._audio_cache or index not in self._waveforms.keys():
			filepath = self.get_audio_fpath(index)
			waveform, _sample_rate = torchaudio.load(filepath)

			if self._audio_cache:
				self._waveforms[index] = waveform

		else:
			waveform = self._waveforms[index]

		if self._audio_transform is not None:
			waveform = self._audio_transform(waveform)
		return waveform

	def get_captions(self, index: int) -> List[str]:
		"""
			:param index: The index of the item.
			:return: The list of 5 captions of an item.
		"""
		info = self._data_info[self.get_audio_fname(index)]
		captions = info['captions']

		if self._captions_transform is not None:
			captions = self._captions_transform(captions)
		return captions

	def get_metadata(self, index: int) -> Dict[str, str]:
		"""
			Returns the metadata dictionary for a file.
			This dictionary contains :
				- 'keywords': Contains the keywords of the item, separated by ';'.
				- 'sound_id': Id of the audio.
				- 'sound_link': Link to the audio.
				- 'start_end_samples': The range of the sound where it was extracted.
				- 'manufacturer': The manufacturer of this file.
				- 'licence': Link to the licence.

			:param index: The index of the item.
			:return: The metadata dictionary associated to the item.
		"""
		info = self._data_info[self.get_audio_fname(index)]
		return info['metadata']

	def get_audio_fname(self, index: int) -> str:
		"""
			:param index: The index of the item.
			:return: The filename associated to the index.
		"""
		return self._idx_to_filename[index]

	def get_audio_fpath(self, index: int) -> str:
		"""
			:param index: The index of the item.
			:return: The filepath associated to the index.
		"""
		info = self._data_info[self.get_audio_fname(index)]
		return info['filepath']

	def get_dataset_root(self) -> str:
		"""
			:return: The folder path of the dataset.
		"""
		return self._dataset_root

	def _download_dataset(self):
		if not osp.isdir(self._dataset_root):
			os.mkdir(self._dataset_root)

		if self._verbose >= 1:
			print('Download files for the dataset...')

		infos = FILES_INFOS[self._subset]

		# Download archives files
		for name, info in infos.items():
			filename, url, hash_ = info['filename'], info['url'], info['hash']
			filepath = osp.join(self._dataset_root, filename)

			if not osp.isfile(filepath):
				if self._verbose >= 1:
					print(f'Download file "{filename}" from url "{url}"...')

				if osp.exists(filepath):
					raise RuntimeError(f'Object "{filepath}" already exists but it\'s not a file.')
				download_url(url, self._dataset_root, filename, hash_value=hash_, hash_type='md5')

		# Extract audio files from archives
		for name, info in infos.items():
			filename = info['filename']
			filepath = osp.join(self._dataset_root, filename)
			extension = filename.split('.')[-1]

			if extension == '7z':
				extracted_path = osp.join(self._dataset_root, self._subset)

				if not osp.isdir(extracted_path):
					if self._verbose >= 1:
						print(f'Extract archive file "{filename}"...')

					archive_file = SevenZipFile(filepath)
					archive_file.extractall(self._dataset_root)
					archive_file.close()

	def _prepare_data(self):
		# Read filepath of .wav audio files
		dirpath_audio = osp.join(self._dataset_root, self._subset)
		self._data_info = {
			filename: {
				'filepath': osp.join(dirpath_audio, filename)
			}
			for filename in os.listdir(dirpath_audio)
		}

		files_infos = FILES_INFOS[self._subset]

		# --- Read captions info
		if self._subset != ClothoV1Subset.TEST:
			captions_filename = files_infos['captions']['filename']
			captions_filepath = osp.join(self._dataset_root, captions_filename)

			# Keys: file_name, caption_1, caption_2, caption_3, caption_4, caption_5
			with open(captions_filepath, 'r') as file:
				reader = csv.DictReader(file)
				for row in reader:
					filename = row['file_name']
					if filename in self._data_info.keys():
						self._data_info[filename]['captions'] = [
							row[caption_key] for caption_key in ['caption_1', 'caption_2', 'caption_3', 'caption_4', 'caption_5']
						]
					else:
						raise RuntimeError(f'Found filename "{filename}" in CSV "{captions_filename}" but not the audio file.')
		else:
			for filename in self._data_info.keys():
				self._data_info[filename]['captions'] = []

		# --- Read metadata info
		metadata_filename = files_infos['metadata']['filename']
		metadata_filepath = osp.join(self._dataset_root, metadata_filename)

		# Keys: file_name, keywords, sound_id, sound_link, start_end_samples, manufacturer, license
		with open(metadata_filepath, 'r') as file:
			reader = csv.DictReader(file)
			for row in reader:
				filename = row['file_name']
				row_copy = dict(row)
				row_copy.pop('file_name')

				if filename in self._data_info.keys():
					self._data_info[filename]['metadata'] = row_copy
				else:
					raise RuntimeError(f'Found filename "{filename}" in CSV "{metadata_filename}" but not the audio file.')

		self._idx_to_filename = [filename for filename in self._data_info.keys()]
