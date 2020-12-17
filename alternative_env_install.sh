#!/bin/sh

conda create -n env_mlu
conda activate env_mlu
conda install pytorch torchaudio torchvision cudatoolkit=10.2 -c pytorch
conda install matplotlib
conda install numpy

pip install nltk

pip install -e .
