# Art-GAN
Generative Adversarial Network for Art Creation

GAN trained on Art of the past Centuries. This projects implements WGAN-GP Architecture for Tensorflow 2.x and is for learning purposes only. \
Read the paper here: [Improved Training of Wasserstein GANs](https://arxiv.org/pdf/1704.00028.pdf).\
Training Set is converted and read via two basic read/write classes.\
Dataset taken from [Kaggle](https://www.kaggle.com/c/painter-by-numbers/overview) 

# Update Log: 
Changed Stride2 Upconvolution to Conv2D with bilinear Upsampling. \
Input Image is downsampled and depth Concatenated to Featuremaps to enforce stronger gradients and low level distinction in the Discriminator. \
BatchNormalization is omitted in the Discriminator and LayerNormalization is used instead. \
Added Gradientpenalty and Wassersteinloss.

The GAN currently struggels to converge and will be updated shortly.

Current Version: \
<img src="https://github.com/smdgn/images/blob/master/wgan2.png" width="224" height="224"> <img src="https://github.com/smdgn/images/blob/master/Screenshot_2020-10-21%20individualImage%20(PNG-Grafik%2C%20448%20%C3%97%20448%20Pixel).png" width="224" height="224">
