import tensorflow as tf
from tensorflow.losses import mean_squared_error as mse
from tensorflow.keras.losses import mean_absolute_error as mae
from tensorflow.keras.losses import BinaryCrossentropy as BCE
#from tensorflow.keras.losses import mae

bce = BCE(from_logits=False, label_smoothing=0.15)

def discriminator_loss(real_output, fake_output):
    real_loss = bce(tf.ones_like(real_output), real_output)
    fake_loss = bce(tf.zeros_like(fake_output), fake_output)
    return real_loss+fake_loss

def generator_loss(fake_output):
    return bce(tf.ones_like(fake_output), fake_output)



