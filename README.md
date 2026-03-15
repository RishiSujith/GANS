# DCGAN Face Generator

*A basic deep learning project built with PyTorch on Google Colab.*

## Overview

This is a very basic implementation of a Deep Convolutional Generative Adversarial Network (DCGAN) that learns to generate human face images from random noise. It was built as a learning exercise and makes no claims of being production-ready or state-of-the-art. The architecture closely follows the original 2015 DCGAN paper by Radford et al.

## What It Does

- Trains a Generator network to produce 64x64 RGB face images from random noise vectors
- Trains a Discriminator network to tell real faces apart from generated ones
- The two networks compete against each other, improving over 5 training epochs
- Saves generated face images and loss curves at the end of training

## Dataset

Trained on the FFHQ (Flickr-Faces-HQ) thumbnails dataset, downloaded via Hugging Face. The dataset contains 70,000 high-quality face images resized to 64x64 for training.

## How It Was Built

- **Environment:** Google Colab with a free T4 GPU
- **Framework:** PyTorch with torchvision
- **Dataset:** Downloaded using the Hugging Face `datasets` library
- **Training time:** roughly 20-30 minutes for 5 epochs on a T4 GPU

## Stack

- Python 3
- PyTorch
- torchvision
- Hugging Face datasets
- matplotlib
- Google Colab

## Limitations

This is intentionally a bare-bones project. It is missing many things a real implementation would have:

- No learning rate scheduling
- No gradient penalty or spectral normalization for training stability
- No FID score or other quantitative evaluation
- Only 5 training epochs results are blurry and inconsistent
- 64x64 output resolution is very low
- Prone to mode collapse if hyperparameters are not tuned carefully

## Results

Generated faces after 5 epochs of training:

![Generated Faces](generated_faces.png)

Loss curves over training:

![Loss Curves](loss_curves.png)