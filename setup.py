#!/usr/bin/python3
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages
from setuptools.command.develop import develop
from setuptools.command.install import install
from subprocess import check_call


install_requires = [
	'torch~=1.7.1',
	'torchaudio~=0.7.2',
	'torchtext~=0.8.1',
	'torchvision~=0.8.2',
	'torchmetrics~=0.2.0',
	'pytorch-lightning~=1.2.3',
	'scikit-learn',
	'tensorboard',
	'nltk',
	'tqdm',
	'matplotlib',
	'numpy',
	'rouge-metric',
	'py7zr',
	'requests',
	'ipython',
	'pillow',
]


class PostDevelopCommand(develop):
	def run(self):
		super().run()
		check_call(['bash', 'install_spice.sh'])


class PostInstallCommand(install):
	def run(self):
		super().run()
		check_call(['bash', 'install_spice.sh'])


setup(
	name='mlu',
	version='0.5.1',
	packages=find_packages(),
	url='https://github.com/Labbeti/MLU',
	license='MIT',
	author='Etienne Labbé',
	author_email='etienne.labbe31@gmail.com',
	description='Set of personal classes, functions and tools for machine learning.',
	python_requires='>=3.8.5',
	install_requires=install_requires,
	include_package_data=True,
	cmdclass={
		'develop': PostDevelopCommand,
		'install': PostInstallCommand,
	}
)
