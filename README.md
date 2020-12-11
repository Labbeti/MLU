# Machine Learning Utils (MLU)

Set of classes, functions and tools for machine learning in Pytorch. 


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
- METEOR,
- Precision,
- Recall

## TODO
- FScore, LCS, NIST, WER metrics.
