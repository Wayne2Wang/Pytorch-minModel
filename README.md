## Pytorch-minModel
This is the repository that holds my implementation for deep learning models. 

Note that this is mainly for educational purposes, so it only contains the minimum code for each model. As can be expected, the hyperparameters are not super optimized and the datasets used are the most accessible(smallest) ones.

##### Implemented :satisfied:

- Generative Adversarial Networks: [GANs](#GANs), [cGANs](#cGANs)
- MLP-like architectures: TODO
- Convolutional Neural Networks: TODO
- Transformers: TODO

##### To be implemented :monocle_face:	
gMLP, ResNet, ViT, StyleGan, CR-GAN, DC-GAN, cDC-GAN

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
- <i>Generative Adversarial Networks (GANs)</i><a id="GANs">
  - paper: https://arxiv.org/abs/1406.2661, 2014
  - author(s): Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, Yoshua Bengio
  - generated images from MNIST and Fashion-MNIST
  <p align="center"><img src="assets/gan.gif" width="300">
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <img src="assets/GAN_49.png" width="300"></p>
    <p align="center"><img src="assets/GAN_fashion.gif" width="300">
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <img src="assets/GAN_49_fashion.png" width="300"></p>
  
  
- <i>Conditional Generative Adversarial Networks (cGANs)</i><a id="cGANs">
  - paper: https://arxiv.org/abs/1411.1784, 2014
  - author(s): Mehdi Mirza, Simon Osindero
  - generated images from MNIST and Fashion-MNIST
  <p align="center"><img width="300" src="assets/cGAN.gif">
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <img width="300" src="assets/cGAN_49.png"></p>
  <p align="center"><img width="300" src="assets/cGAN_fashion.gif">
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <img width="300" src="assets/cGAN_49_fashion.png"></p>
