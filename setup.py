#!/usr/bin/python3
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages


install_requires = ["torch", "torchaudio", "torchvision", "nltk", "matplotlib", "numpy"]


setup(
	name="mlu",
	version="0.3",
	packages=find_packages(),
	url="https://github.com/Labbeti/MLU",
	license="MIT",
	author="Etienne Labbe 'Labbeti'",
	author_email="etienne.labbe31@gmail.com",
	description="Set of personal classes, functions and tools for machine learning.",
	install_requires=install_requires,
)
