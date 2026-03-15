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

 starting training loop
[1/5][0/547] Loss_D: 1.3734  Loss_G: 5.0239
[1/5][50/547] Loss_D: 2.8131  Loss_G: 24.6011
[1/5][100/547] Loss_D: 0.9989  Loss_G: 10.9284
[1/5][150/547] Loss_D: 0.9862  Loss_G: 4.3691
[1/5][200/547] Loss_D: 0.1943  Loss_G: 4.3848
[1/5][250/547] Loss_D: 1.9501  Loss_G: 3.4334
[1/5][300/547] Loss_D: 0.6165  Loss_G: 3.7714
[1/5][350/547] Loss_D: 0.5091  Loss_G: 4.3562
[1/5][400/547] Loss_D: 0.8902  Loss_G: 6.5919
[1/5][450/547] Loss_D: 0.7638  Loss_G: 5.1201
[1/5][500/547] Loss_D: 0.5461  Loss_G: 2.9902
[2/5][0/547] Loss_D: 0.3881  Loss_G: 4.5606
[2/5][50/547] Loss_D: 0.9705  Loss_G: 4.1659
[2/5][100/547] Loss_D: 0.3704  Loss_G: 4.4554
[2/5][150/547] Loss_D: 1.3227  Loss_G: 14.0933
[2/5][200/547] Loss_D: 0.5130  Loss_G: 7.1236
[2/5][250/547] Loss_D: 0.3087  Loss_G: 5.7186
[2/5][300/547] Loss_D: 0.3529  Loss_G: 4.8853
[2/5][350/547] Loss_D: 0.3673  Loss_G: 4.7858
[2/5][400/547] Loss_D: 0.4590  Loss_G: 5.3112
[2/5][450/547] Loss_D: 0.4110  Loss_G: 3.3709
[2/5][500/547] Loss_D: 0.4438  Loss_G: 6.7803
[3/5][0/547] Loss_D: 0.8441  Loss_G: 4.6333
[3/5][50/547] Loss_D: 0.9050  Loss_G: 6.1149
[3/5][100/547] Loss_D: 0.9729  Loss_G: 1.5587
[3/5][150/547] Loss_D: 0.4496  Loss_G: 3.8696
[3/5][200/547] Loss_D: 0.5839  Loss_G: 4.7663
[3/5][250/547] Loss_D: 0.4791  Loss_G: 3.6286
[3/5][300/547] Loss_D: 0.5904  Loss_G: 3.2327
[3/5][350/547] Loss_D: 0.6780  Loss_G: 2.8814
[3/5][400/547] Loss_D: 0.5056  Loss_G: 3.5239
[3/5][450/547] Loss_D: 0.5753  Loss_G: 3.9606
[3/5][500/547] Loss_D: 0.5730  Loss_G: 3.1465
[4/5][0/547] Loss_D: 0.6655  Loss_G: 3.0861
[4/5][50/547] Loss_D: 0.6860  Loss_G: 5.5584
[4/5][100/547] Loss_D: 2.1595  Loss_G: 9.2161
[4/5][150/547] Loss_D: 0.5852  Loss_G: 1.8479
[4/5][200/547] Loss_D: 0.9828  Loss_G: 1.8554
[4/5][250/547] Loss_D: 0.3202  Loss_G: 4.3436
[4/5][300/547] Loss_D: 0.4932  Loss_G: 3.6679
[4/5][350/547] Loss_D: 0.4096  Loss_G: 2.8835
[4/5][400/547] Loss_D: 1.3871  Loss_G: 2.3602
[4/5][450/547] Loss_D: 1.5275  Loss_G: 6.9753
[4/5][500/547] Loss_D: 0.2304  Loss_G: 4.7687
[5/5][0/547] Loss_D: 0.3709  Loss_G: 4.2047
[5/5][50/547] Loss_D: 0.3805  Loss_G: 4.1901
[5/5][100/547] Loss_D: 0.4431  Loss_G: 3.3678
[5/5][150/547] Loss_D: 0.3654  Loss_G: 3.6005
[5/5][200/547] Loss_D: 0.5906  Loss_G: 3.5151
[5/5][250/547] Loss_D: 0.4054  Loss_G: 3.0200
[5/5][300/547] Loss_D: 1.0403  Loss_G: 1.0231
[5/5][350/547] Loss_D: 0.5547  Loss_G: 2.4855
[5/5][400/547] Loss_D: 0.3798  Loss_G: 4.1571
[5/5][450/547] Loss_D: 0.7903  Loss_G: 4.4832
[5/5][500/547] Loss_D: 0.9124  Loss_G: 2.3451

Training done!

![alt text](image.png)
![alt text](image-1.png)