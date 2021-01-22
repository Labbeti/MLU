# Machine Learning Utils (MLU)

Set of classes, functions and tools for machine learning in Pytorch.

## Installation
- Clone the repository :
```bash
pip install git+https://github.com/Labbeti/MLU
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
