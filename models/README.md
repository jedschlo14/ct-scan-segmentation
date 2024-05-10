# Models

This folder contains segmentation models for 3D images. Below is a brief description of each model.

### PatchModel

`PatchModel` is a parent class of every model in this repository. This keeps track of the patch size used and has an inference method.

### BaselineModel

This idea of `BaselineModel` is simplicity. Convolution, max pool, transposed convolution, then a final convolution.

### ExperimentalModel

`ExperimentalModel` builds off of `BaselineModel`. It adds on additional convolutions and channels.

### VolumetricFCN

`VolumetricFCN` is inspired by a [Fully Convolutional Network](https://arxiv.org/abs/1411.4038). It is modified to work on 3D images, and it does not follow the exact architecture.

### U-Net

This model is a U-Net that is modified to work on 3D images. The code for this model is sourced from [Pytorch-UNet](https://github.com/milesial/Pytorch-UNet).
