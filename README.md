## Pytorch-minModel
This is the repository that holds my implementation for deep learning models. 

Note that this is mainly for educational purposes, so it only contains the minimum code for each model. As can be expected, the hyperparameters are not super optimized and the datasets used are the most accessible(smallest) ones.

## Quick start
To set up the right environment, run the following
```
pip install -r requirements.txt
```
(Optional) If you have an NVIDIA GPU, you can accelerate training by installing [CUDA](https://developer.nvidia.com/cuda-downloads/) and [cuDNN](https://developer.nvidia.com/cudnn) of the correct version.

Typically, one only needs to go to the folder for the desired model and run 
```
python train.py
```
Specific instructions will be provided in the "Implemented models" section, if needed.


## Implemented models
- <i>Generative Adversarial Networks (GANs)</i>
  - paper: https://arxiv.org/abs/1406.2661, 2014
  - author(s): Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, Yoshua Bengio
  - generated images on MNIST
  <p align="center"><img src="assets/gan.gif" width="300">
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <img src="assets/GAN_49.png" width="300"></p>
  
  
- <i>Conditional Generative Adversarial Networks (cGANs)</i>
  - paper: https://arxiv.org/abs/1411.1784, 2014
  - author(s): Mehdi Mirza, Simon Osindero
  - generated images on MNIST
  <p align="center"><img width="350" src="assets/cGAN.gif">
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <img width="350" src="assets/cGAN_49.png"></p>

## To be implemented
- gMLP
- ResNet
- ViT
- StyleGan
