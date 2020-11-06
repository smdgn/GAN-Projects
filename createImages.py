import generator
import tensorflow as tf
import matplotlib.pyplot as plt
import utils
import numpy as np

def interpolate(z1, z2, n_steps=10):
    ratios = np.linspace(0, 1, n_steps)
    vectors = [(1.0 - ratio) * z1 + ratio * z2 for ratio in ratios]
    return np.array(vectors)

def interpolate_batch(z, n_steps=10):
    vectors = [interpolate(z1, z2, n_steps) for z1, z2 in zip(z, z[1:])]
    return np.array(vectors)

def norm(image):
    image = tf.add(image, float(1))
    image = tf.multiply(image, float(0.5))
    return image

model_path =r"C:\Users\samed\Google Drive\Art_Net\Model"
batch_size= 1
lr = 1e-4
steps = 15

img_generator = generator.Generator()
discriminator = generator.Discriminator()

generator_optimizer = tf.keras.optimizers.Adam(learning_rate=lr, beta_1 = 0.5, beta_2=0.9)  #Wgan gp uses adam
discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=lr, beta_1 = 0.5, beta_2=0.9)
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer, discriminator_optimizer=discriminator_optimizer, img_generator=img_generator, discriminator=discriminator)
checkpoint.restore(tf.train.latest_checkpoint(model_path))

noise = tf.random.uniform([batch_size, 100]) #latent noise vector
#one =tf.ones_like(noise)
fak = img_generator(noise, training = False)
plt.imshow(norm(fak)[0, :, :, :])
plt.show()


if(0):
    
    samples = interpolate_batch(noise, steps)

    fake_images = [utils.cast_im(norm(img_generator(sample, training=False))) for sample in samples]

    fig = plt.figure()

    for i in range(batch_size-1):
        for j in range(steps):
            plt.imshow(fake_images[i][ j, :, :, :])
            plt.axis('off')
            plt.savefig(r'C:\Users\samed\OneDrive\Bilder\AI\image{:02d}_from_batch{:02}'.format(i, j))
    #plt.show()

