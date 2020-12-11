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
- Normalize,
- Standardize,
- Gray,
- CutOutImg,
- UniColor,
- Inversion

### Spectrogram transforms
- CutOutSpec

### Waveform transforms
- Crop,
- Occlusion,
- Pad,
- PadCrop

## Metrics
- Precision,
- Recall,
- BLEU,
- METEOR

## TODO
- FScore, LCS, NIST, WER metrics.
