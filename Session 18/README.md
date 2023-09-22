# Session 18

# UNETs, Variational AutoEncoders, and Applications

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![PyTorch 2.0](https://img.shields.io/badge/torch-v2.0-brightgreen)](https://pytorch.org/docs/stable/index.html)
[![PyTorch-Lightning](https://img.shields.io/badge/pytorch_lightning-v2.0.6-orange)](https://lightning.ai/docs/pytorch/latest/)
[![PyTorch-Lightning-Bolts](https://img.shields.io/badge/pytorch_lightning_bolts-0.3.2.post1-red)](https://lightning.ai/docs/pytorch/stable/ecosystem/bolts.html)
[![Torchvision 0.15](https://img.shields.io/badge/torchvision-v0.15-green)](https://pytorch.org/vision/stable/index.html)
[![Albumentations](https://img.shields.io/badge/Albumentations-v1.3.1-yellow)](https://albumentations.ai/docs/)

<br>

# Task

## PART 1

1. First part of your assignment is to train your own UNet from scratch, you can use the dataset and strategy provided. However, you need to train it 4 times:

- MP+Tr+BCE
- MP+Tr+Dice Loss
- StrConv+Tr+BCE
- StrConv+Ups+Dice Loss   
and report your results.

<br>

## PART 2

2. Design a variation of a VAE that:

- takes in two inputs:
    - an MNIST image, and
    - its label (one hot encoded vector sent through an embedding layer)
- Training as you would train a VAE
- Now randomly send an MNIST image, but with a wrong label. Do this 25 times, and share what the VAE makes (25 images stacked in 1 image)!
- Now do this for CIFAR10 and share 25 images (1 stacked image)!

<br>

# Solution

This repository contains **2 Parts** corresponding to the given tasks.

<br>

# Image Segmentation

![Image Segmentation](../Results/Session%2018/image_seg.png)

<br>

# UNET

- UNET is a CNN architecture commonly used for image segmentation tasks. 
- It was introduced by Olaf Ronneberger, Philipp Fischer, and Thomas Brox in their 2015 paper titled "U-Net: Convolutional Networks for Biomedical Image Segmentation." 
- UNET is particularly popular in biomedical image analysis, but it has also found applications in various other domains, such as satellite image segmentation, autonomous driving, and more.
- It consists of a contracting path, which gradually reduces the spatial resolution of the input image while increasing the number of channels, and an expanding path, which gradually recovers the original resolution while decreasing the number of channels. 
- The two paths are connected by skip connections, which allow the network to use information from the contracting path to better localize the segmentation in the expanding path. 

![UNET](../Results/Session%2018/unet.png)

![Encoder Decoder](../Results/Session%2018/unet_enc_dec.png)

<br>

# Dice Loss

```python
def dice_loss(pred, target):
    smooth = 1e-5
    
    # flatten predictions and targets
    pred = pred.view(-1)
    target = target.view(-1)
    
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    
    dice = (2. * intersection + smooth) / (union + smooth)
    
    return 1 - dice  
```

<br>

# AutoEncoders

- Autoencoders are a type of neural network architecture used for unsupervised learning, where the goal is to learn a compressed representation (encoding) of the input data. 
- The basic idea behind autoencoders is to learn a function that maps the input data to a lower-dimensional representation and then reconstructs the original data from the encoded representation.

![Autoencoders](../Results/Session%2018/autoencoders.png)

<br>

# Variational Autoencoders

- Variational Autoencoders (VAEs) have one fundamentally unique property that separates them from vanilla autoencoders, and it is this property that makes them so useful for generative modeling: their latent spaces are, by design, continuous, allowing easy random sampling and interpolation. 
- It achieves this by doing something that seems rather surprising at first: making its encoder NOT output an encoding vector of size n, but rather, outputting two vectors of size n: a vector of means μ, and another vector of standard deviations σ!

![VAE](../Results/Session%2018/vae.png)

<br>

# Kullback-Leibler Divergence

- The KL divergence between two probability distributions simply measures how much they diverge from each other. 
- Minimizing the KL divergence here means optimizing the probability distribution parameters (μ and σ) to closely resemble that of the target distribution. 
- **KL divergence is minimized when μ = 0 and σ = 1**

$$
KL(P \Vert Q) = \sum_{x=1}^{n} P(x) \cdot \log\left(\frac{P(x)}{Q(x)}\right)
$$

<br>