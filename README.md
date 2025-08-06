# Adaptive Token Dictionary

This repository is an official implementation of the paper "Towards Efficient Image Super-Resolution via Structured Sparse and Local Representation Learning".


## Environment
- Python 3.9
- PyTorch 2.0.1

## Training
### Data Preparation
- Download the training dataset DF2K ([DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) + [Flickr2K](https://cv.snu.ac.kr/research/EDSR/Flickr2K.tar)) and put them in the folder `./datasets`.
- It's recommanded to refer to the data preparation from [BasicSR](https://github.com/XPixelGroup/BasicSR/blob/master/docs/DatasetPreparation.md) for faster data reading speed.

### Training Commands
# ×4 
CUDA_VISIBLE_DEVICES=0 python basicsr/train.py -opt options/train/SRx4.yml --launcher none
```

## Testing
### Data Preparation
- Download the testing data (Set5 + Set14 + BSD100 + Urban100 + Manga109 [[download](https://drive.google.com/file/d/1_FvS_bnSZvJWx9q4fNZTR8aS15Rb0Kc6/view?usp=sharing)]) and put them in the folder `./datasets`.

### Pretrained Models
- Download the [pretrained models](https://drive.google.com/drive/folders/1D3BvTS1xBcaU1mp50k3pBzUWb7qjRvmB?usp=sharing) and put them in the folder `./experiments/pretrained_models`.

### Testing Commands
- Refer to the testing configuration files in `./options/test` folder for detailed settings.
- ATD (Classical Image Super-Resolution)
```bash
python basicsr/test.py -opt options/test/001_ATD_SRx2_finetune.yml
python basicsr/test.py -opt options/test/002_ATD_SRx3_finetune.yml
python basicsr/test.py -opt options/test/003_ATD_SRx4_finetune.yml
```

- ATD-light (Lightweight Image Super-Resolution)
```bash
python basicsr/test.py -opt options/test/101_ATD_light_SRx2_scratch.yml
python basicsr/test.py -opt options/test/102_ATD_light_SRx3_finetune.yml
python basicsr/test.py -opt options/test/103_ATD_light_SRx4_finetune.yml
```


## Results
- Classical Image Super-Resolution

<img width="800" src="figures/classical.png">

- Lightweight Image Super-Resolution

<img width="800" src="figures/lightweight.png">

## Visual Results

- Complete visual results can be downloaded from [link](https://drive.google.com/drive/folders/1HwEbAGU6WEw9ZGbFdt__BOJo_5DKflEb?usp=sharing).

<img width="800" src="figures/viscomp_027.png">
<img width="800" src="figures/viscomp_momo.png">


## Citation

```
@InProceedings{Zhang_2024_CVPR,
    author    = {Zhang, Leheng and Li, Yawei and Zhou, Xingyu and Zhao, Xiaorui and Gu, Shuhang},
    title     = {Transcending the Limit of Local Window: Advanced Super-Resolution Transformer with Adaptive Token Dictionary},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2024},
    pages     = {2856-2865}
}
```

## Acknowledgements
This code is built on [BasicSR](https://github.com/XPixelGroup/BasicSR).

