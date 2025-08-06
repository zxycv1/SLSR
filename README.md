# SLSR

This repository is an official implementation of the paper "Towards Efficient Image Super-Resolution via Structured Sparse and Local Representation Learning".


## Environment
- Python 3.9
- PyTorch 2.0.1

## Training
### Data Preparation
- Download the training dataset DF2K ([DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) + [Flickr2K](https://cv.snu.ac.kr/research/EDSR/Flickr2K.tar)) and put them in the folder `./datasets`.
- It's recommanded to refer to the data preparation from [BasicSR](https://github.com/XPixelGroup/BasicSR/blob/master/docs/DatasetPreparation.md) for faster data reading speed.

### Training Commands
```bash
CUDA_VISIBLE_DEVICES=0 python basicsr/train.py -opt options/train/SRx4.yml --launcher none
```

## Testing
### Data Preparation
- Download the testing data (Set5 + Set14 + BSD100 + Urban100 + Manga109 [[download](https://drive.google.com/file/d/1_FvS_bnSZvJWx9q4fNZTR8aS15Rb0Kc6/view?usp=sharing)]) and put them in the folder `./datasets`.

```bash
python basicsr/test.py -opt options/test/101_ATD_light_SRx2_scratch.yml
python basicsr/test.py -opt options/test/102_ATD_light_SRx3_finetune.yml
python basicsr/test.py -opt options/test/SRx4.yml
```


## Acknowledgements
This code is built on [BasicSR](https://github.com/XPixelGroup/BasicSR).

