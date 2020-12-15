# Machine Learning Utils (MLU)

Set of classes, functions and tools for machine learning in Pytorch.

## Installation
- Clone the repository :
```bash
git clone https://github.com/Labbeti/MLU
```
- Create a conda environment with the YAML file (passwords can be required during installation) :
```bash
cd MLU
conda env create -f environment.yml
```
- Activate the new environment :
```bash
conda activate env_mlu
```
- Setup the main repository :
```bash
pip install -e .
```

# Content
## Neural Network utils
### Losses
- CrossEntropyWithVectors,
- Entropy,
- JSDivLoss,
- KLDivLossWithProbabilities

### Others
- Squeeze,
- UnSqueeze,
- OneHot,
- Mish

## Schedulers
- CosineLRScheduler,
- SoftCosineLRScheduler

## Transforms / Augmentations
### Image transforms
#### For PIL images
- AutoContrast,
- Brightness,
- Color,
- Contrast,
- CutOutImgPIL  
- Equalize,
- HorizontalFlip,
- IdentityImage,
- Invert,
- Posterize,
- RandAugment,
- Rescale,
- Rotation,
- Sharpness,
- ShearX,
- ShearY,
- Smooth,
- Solarize,
- TranslateX,
- TranslateY,
- VerticalFlip

#### For tensors images
- CutOutImg,
- Gray,
- Inversion,
- Normalize,
- Standardize,
- UniColor

### Spectrogram transforms
- CutOutSpec

### Waveform transforms
- Crop,
- Occlusion,
- Pad,
- PadCrop,
- TimeStretchNearest,
- StretchPadCrop

## Metrics
- BLEU,
- CategoricalAccuracy,
- FScore,
- LCS,
- METEOR,
- NIST,
- Precision,
- Recall,
- WER

## TODO
- ROUGE, SPICE and SPIDEr metrics
