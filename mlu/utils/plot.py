
import torch

from matplotlib import pyplot as plt
from torch import Tensor
from typing import Optional


def plot_waveform(
	waveform: Tensor,
	sample_rate: int,
	title: str = "Waveform",
	xlim: Optional[int] = None,
	ylim: Optional[int] = None,
):
	"""
		BASED ON PYTORCH AUTIO TUTORIAL :
		https://pytorch.org/tutorials/beginner/audio_preprocessing_tutorial.html
	"""
	waveform = waveform.numpy()

	num_channels, num_frames = waveform.shape
	time_axis = torch.arange(0, num_frames) / sample_rate

	figure, axes = plt.subplots(num_channels, 1)
	if num_channels == 1:
		axes = [axes]
	for c in range(num_channels):
		axes[c].plot(time_axis, waveform[c], linewidth=1)
		axes[c].grid(True)
		if num_channels > 1:
			axes[c].set_ylabel(f'Channel {c + 1}')
		if xlim:
			axes[c].set_xlim(xlim)
		if ylim:
			axes[c].set_ylim(ylim)
	figure.suptitle(title)
	plt.show(block=False)


def plot_specgram(
	waveform: Tensor,
	sample_rate: int,
	title: str = "Spectrogram",
	xlim: Optional[int] = None,
):
	"""
		BASED ON PYTORCH AUTIO TUTORIAL :
		https://pytorch.org/tutorials/beginner/audio_preprocessing_tutorial.html
	"""
	waveform = waveform.numpy()

	num_channels, num_frames = waveform.shape
	# time_axis = torch.arange(0, num_frames) / sample_rate

	figure, axes = plt.subplots(num_channels, 1)
	if num_channels == 1:
		axes = [axes]
	for c in range(num_channels):
		axes[c].specgram(waveform[c], Fs=sample_rate)
		if num_channels > 1:
			axes[c].set_ylabel(f'Channel {c + 1}')
		if xlim:
			axes[c].set_xlim(xlim)
	figure.suptitle(title)
	plt.show(block=False)