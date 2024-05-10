# CT Scan Segmentation

This repository contains code developed for [PENGWIN](https://pengwin.grand-challenge.org/) (Pelvic Bone Fragments with Injuries Segmentation Challenge), which focuses on segmenting CT scans. The main objective is to develop segmentation models for CT scans, particularly focusing on pelvic bone fragments with injuries. The data used for analysis is sourced from [https://zenodo.org/records/10927452](https://zenodo.org/records/10927452).

## Disclaimer

This project is developed as a final project for BMI 567 (Medical Image Analysis) at UW-Madison. As a result, there was time and hardware resources, so this repository does not aim to achieve state-of-the-art performance. Instead, it serves as a template for working with 3D medical images and implementing segmentation algorithms.


## Directory Structure
- `data/`: Contains model weights and data such as training losses.
- `models/`: Contains implementations of segmentation models.
- `utils/`: Contains utility functions used across the project.


## Usage

### Installation:
Clone the repository

```
git clone https://github.com/jedschlo14/ct-scan-segmentation.git
```

### Create a Dataset:
To create a dataset, format data with the following structure:

```
dataset/
├── test/
│   ├── image/
│   │   ├── test_1.mha
│   │   ├── test_2.mha
│   │   └── ...
│   └── mask/
│       ├── test_1.mha
│       ├── test_2.mha
│       └── ...
├── train/
│   ├── image/
│   │   ├── train_1.mha
│   │   ├── train_2.mha
│   │   └── ...
│   └── mask/
│       ├── train_1.mha
│       ├── train_2.mha
│       └── ...
└── val/
    ├── image/
    │   ├── val_1.mha
    │   ├── val_2.mha
    │   └── ...
    └── mask/
        ├── val_1.mha
        ├── val_2.mha
        └── ...
```

Note that this repository is designed for work with `.mha` files, but it may work fine with other extensions supported by [TorchIO](https://torchio.readthedocs.io/).

### Training:

You can train segmentation models using `train.py`. Training weights and losses will be stored in `data/<model_name>`.

Example:
```
python3 train.py --device cuda --model BaselineModel --model_name BaselineModel --batch_size 2048 --num_classes 2 --dataset_path 'datasets/pengwin'
```

See all options with
```
python3 train.py --h
```

### Evaluation:

You can evaluate segmentation models using `evaluate.py`. Dice and IoU coefficients will be stored in `data/<model_name>`.

Example:
```
python3 evaluate.py --device cuda --model BaselineModel --model_name BaselineModel --batch_size 2048 --num_classes 2 --dataset_path 'datasets/pengwin'
```

See all options with
```
python3 evaluate.py --h
```

## Acknowledgments

- The authors of the code used in the `models/unet` folder. The code was originally sourced from [Pytorch-UNet](https://github.com/milesial/Pytorch-UNet).
- The authors of `utils/diceLoss.py`. This code was sourced from [weiliu620](https://gist.github.com/weiliu620/52d140b22685cf9552da4899e2160183)
