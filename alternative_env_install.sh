#!/bin/sh

conda create -n env_mlu
conda activate env_mlu
conda install pytorch cudatoolkit=10.2 -c pytorch
conda install matplotlib

pip install nltk

pip install -e .
