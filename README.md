# CT Scan Segmentation

This repository contains code developed for [PENGWIN](https://pengwin.grand-challenge.org/) (Pelvic Bone Fragments with Injuries Segmentation Challenge), which focuses on segmenting CT scans. The main objective is to develop segmentation models for CT scans, particularly focusing on pelvic bone fragments with injuries. 

## Disclaimer

This project is developed as a final project for BMI 567 (Medical Image Analysis) at UW-Madison. As a result, there was time and hardware resources, so this repository does not aim to achieve state-of-the-art performance. Instead, it serves as a template for working with 3D medical images and implementing segmentation algorithms.


## Directory Structure
- `datasets/`: Contains datasets for training and evaluation.
- `data/`: Contains model weights and data such as training losses.
- `models/`: Contains implementations of segmentation models.
- `utils/`: Contains utility functions used across the project.


## Acknowledgments

- The authors of the code used in the `models/unet` folder. The code was originally sourced from [Pytorch-UNet](https://github.com/milesial/Pytorch-UNet).
