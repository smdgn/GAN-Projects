import tensorflow as tf

from tensorflow.keras.losses import BinaryCrossentropy as BCE
#from tensorflow.keras.losses import mae

bce = BCE(from_logits=False, label_smoothing=0.15)

def discriminator_loss_bce(real_output, fake_output):   #DCGAN Loss
    real_loss = bce(tf.ones_like(real_output), real_output)
    fake_loss = bce(tf.zeros_like(fake_output), fake_output)
    return real_loss+fake_loss

def generator_loss_bce(fake_output):
    return bce(tf.ones_like(fake_output), fake_output)


def discriminator_loss_w(real_output, fake_output):   #WGAN Loss *Wasserstein distance*
    return tf.reduce_mean(fake_output) - tf.reduce_mean(real_output)

def generator_loss_w(fake_output):
    return -tf.reduce_mean(fake_output)

def gradient_penalty(fake, real, discriminator):
    b, w, h, c = real.shape
    #fake = tf.image.resize(fake, [w, h])
    alpha = tf.random.uniform([b, 1, 1, 1], 0., 1.)
    interpolate = real + (alpha * (fake- real))
    with tf.GradientTape(persistent=True) as grad:
        grad.watch(interpolate)
        prediction = discriminator(interpolate, training=True)
    gradient = grad.gradient(prediction, [interpolate])[0]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradient), axis = [1,2,3]))
    penalty_loss = tf.reduce_mean((slopes-1.)**2)
    return penalty_loss