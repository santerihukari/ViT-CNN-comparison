# ViT-CNN comparison

Implementation of Bachelor's thesis experimental part of Santeri Hukari started in September 2023. 

This project uses PyTorch Lightning 2.10 
## Credits
Pytorch Lightning 2.10 tutorial 11: Vision Transformers by Phillip Lippe was used as a codebase that went under heavy alterations. 

## Models
A Vision Transformer based network shown in Lippe's tutorial was used 

## Datasets
Datasets used in experiments include
- CIFAR-10

## Subsets
TODO: explain methodology for subsampling and subsample sizes, randomness etc.

## Environment
Miniconda 23.10.0 was used to test this pipeline on a i7 cpu with no gpu acceleration available.

python=3.11.5

By default as of 7.11.2023 conda offers lightning versions up to 2.0.3 along with other older packages. 
With the lines below you will get the right packages.
### Manual setup
//TODO: add package versions in the install commands

Starting from a fresh miniconda install the following procedures were followed to get the project running.

```
conda create -n ViTCNN python=3.11.5
conda activate ViTCNN
conda install --name ViTCNN lightning -c conda-forge
conda install --name ViTCNN -c pytorch-nightly torchvision
conda install --name ViTCNN -c conda-forge tensorboard
```
```
Package name                    Version
pytorch-lightning               2.1.0
torchvision                     0.16.0
tensorboard                     2.15.1
```
### Conda export file

## Training


## Inference

The results are saved as tensorboards saved in "tb_logs/{obj_name}_{exp_name}_predict". To see the results run:
TODO: how to run tensorboard and see results
