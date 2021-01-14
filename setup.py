#!/usr/bin/python3
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages


install_requires = [
	"torch==1.7.0",
	"torchaudio==0.7.0",
	"torchvision==0.8.1",
	"tensorboard",
	"nltk",
	"matplotlib",
	"numpy"
]


setup(
	name="mlu",
	version="0.2.3",
	packages=find_packages(),
	url="https://github.com/Labbeti/MLU",
	license="MIT",
	author="Etienne Labbe 'Labbeti'",
	author_email="etienne.labbe31@gmail.com",
	description="Set of personal classes, functions and tools for machine learning.",
	install_requires=install_requires,
	include_package_data=True,
)
