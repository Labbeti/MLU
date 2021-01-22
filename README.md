# Machine Learning Utils (MLU)

Set of classes, functions and tools for machine learning in Pytorch.

## Installation
- Run this command in your environment :
```bash
pip install git+https://github.com/Labbeti/MLU
```

## Requirements
- python>=3.8.5,
- torch==1.7.0,
- torchaudio==0.7.0, 
- torchvision==0.8.1,
- tensorboard>=2.4.0,
- nltk>=3.5,
- matplotlib>=3.3.2,
- numpy>=1.19.2
- rouge-metric>=1.0.1

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
- Mish,
- EMA (Exponential Moving Average of modules),

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
- RollSpec,

### Waveform transforms
- Crop,
- Occlusion,
- Pad,
- TimeStretchNearest,
- StretchPadCrop

### Other transforms
- ToTensor, ToNumpy, ToList, ToPIL,
- Compose, RandomChoice

## Metrics
### Classification
- Average Precision (AP),
- BinaryAccuracy,  
- CategoricalAccuracy,
- D-prime,
- FScore,
- Precision,
- Recall,
- RocAuc (AUC),

### Translation
- BLEU,
- LCS,
- METEOR,
- NIST,
- Rouge-L  
- WER

## Utilities
- ColumnPrinter, LinePrinter,
- ZipCycle

## TODO
- Other ROUGEs, SPICE and SPIDEr metrics
