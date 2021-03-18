#!/usr/bin/python3
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages


install_requires = [
	# "torch~=1.8.0",
	# "torchaudio~=0.8.0",
	# "torchtext~=0.9.0",
	# "torchvision~=0.9.0",
	"torch~=1.7.1",
	"torchaudio~=0.7.2",
	"torchtext~=0.8.1",
	"torchvision~=0.8.2",
	"pytorch-lightning~=1.2.3",
	"tensorboard",
	"nltk",
	"matplotlib",
	"numpy",
	"rouge-metric",
]


setup(
	name="mlu",
	version="0.4.3",
	packages=find_packages(),
	url="https://github.com/Labbeti/MLU",
	license="MIT",
	author="Etienne Labbe 'Labbeti'",
	author_email="etienne.labbe31@gmail.com",
	description="Set of personal classes, functions and tools for machine learning.",
	python_requires=">=3.8.5",
	install_requires=install_requires,
	include_package_data=True,
)
