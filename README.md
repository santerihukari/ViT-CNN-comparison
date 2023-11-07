# ViT-CNN comparison

Implementation of Bachelor's thesis experimental part of Santeri Hukari started in September 2023. 

This project uses PyTorch Lightning 2.10 

# Quick setup

```
conda create -n ViTCNN --file env.txt python=3.11.5
conda activate ViTCNN
python main.py
```
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
Miniconda 23.10.0 was used to test this pipeline on a i7 cpu with no gpu acceleration available with python version 3.11.5.

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
The file [env.txt](env.txt) contains packages that have been tested to 
work with training the network in a WSL2 system running ubuntu on 
Microsoft Windows 11 Pro 

## Conda import
```
conda create -n ViTCNN python=3.11.5
conda create -n ViTCNN --file env.txt python=3.11.5

conda activate ViTCNN
conda
```



## Test hardware
```
(ViTCNN) santeri@LAPTOP-67I7VNKD:~/ViT-CNN-comparison$ lsb_release -a
No LSB modules are available.
Distributor ID: Ubuntu
Description:    Ubuntu 22.04.3 LTS
Release:        22.04
Codename:       jammy

(ViTCNN) santeri@LAPTOP-67I7VNKD:~/ViT-CNN-comparison$ lscpu
Architecture:            x86_64
  CPU op-mode(s):        32-bit, 64-bit
  Address sizes:         39 bits physical, 48 bits virtual
  Byte Order:            Little Endian
CPU(s):                  8
  On-line CPU(s) list:   0-7
Vendor ID:               GenuineIntel
  Model name:            Intel(R) Core(TM) i7-10510U CPU @ 1.80GHz
    CPU family:          6
    Model:               142
    Thread(s) per core:  2
    Core(s) per socket:  4
    Socket(s):           1
    Stepping:            12
    BogoMIPS:            4608.01
    Flags:               fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflu
                         sh mmx fxsr sse sse2 ss ht syscall nx pdpe1gb rdtscp lm constant_tsc rep_good
                          nopl xtopology cpuid pni pclmulqdq vmx ssse3 fma cx16 pcid sse4_1 sse4_2 mov
                         be popcnt aes xsave avx f16c rdrand hypervisor lahf_lm abm 3dnowprefetch invp
                         cid_single ssbd ibrs ibpb stibp ibrs_enhanced tpr_shadow vnmi ept vpid ept_ad
                          fsgsbase bmi1 avx2 smep bmi2 erms invpcid rdseed adx smap clflushopt xsaveop
                         t xsavec xgetbv1 xsaves flush_l1d arch_capabilities
```
## Training


## Inference

The results are saved as tensorboards saved in "tb_logs/{obj_name}_{exp_name}_predict". To see the results run:
TODO: how to run tensorboard and see results
