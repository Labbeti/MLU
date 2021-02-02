#!/usr/bin/python3
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages


install_requires = [
	"torch~=1.7.0",
	"torchaudio~=0.7.0",
	"torchvision~=0.8.1",
	"tensorboard",
	"nltk",
	"matplotlib",
	"numpy",
	"rouge-metric"
]


setup(
	name="mlu",
	version="0.3.1",
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
