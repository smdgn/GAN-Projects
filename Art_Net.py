<<<<<<< HEAD
import art_datareader
import matplotlib.pyplot as plt
import numpy as np
import generator
import loss_utils
import utils
import os
import sys

import tensorflow as tf
from tensorboard.plugins.beholder import Beholder
import tensorflow.summary as summary

#resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='grpc://' + os.environ['COLAB_TPU_ADDR'])
#tf.config.experimental_connect_to_cluster(resolver)
# This is the TPU initialization code that has to be at the beginning.
#tf.tpu.experimental.initialize_tpu_system(resolver)
#print("All devices: ", tf.config.list_logical_devices('TPU'))


epochs = int(100)
batch_size = 8
lr = 1e-4
gp_lamda = 10.
#dir = "C:\\Users\\samed\\OneDrive\\Dokumente\\Datasets\\ArtDataset\\TFrecord_train"
#logs_path = "C:\\Users\\samed\\OneDrive\\Dokumente\\Datasets\\Logs" 

directory = "/content/drive/My Drive/Art_Net/Dataset"
logs_path= "/content/drive/My Drive/Art_Net/Logs"
model_path ="/content/drive/My Drive/Art_Net/Model"

if __name__ == '__main__':
    #session_name = utils.get_session_name()
    session_logs_path = logs_path + "/" + "Test2"

    datareader = art_datareader.DataReader(batch_size, directory)
    img_generator = generator.Generator()
    discriminator = generator.Discriminator()

    #generator_optimizer = tf.keras.optimizers.RMSprop(learning_rate=lr)  #Wgan uses RMSprop
    #discriminator_optimizer = tf.keras.optimizers.RMSprop(learning_rate=lr) 

    generator_optimizer = tf.keras.optimizers.Adam(learning_rate=lr, beta_1 = 0.5, beta_2=0.9)  #Wgan gp uses adam
    discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=lr, beta_1 = 0.5, beta_2=0.9)

    writer = tf.summary.create_file_writer(session_logs_path, max_queue=0)
    checkpoint_prefix = os.path.join(model_path, "ckpt")
    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer, discriminator_optimizer=discriminator_optimizer, img_generator=img_generator, discriminator=discriminator)
    #checkpoint.restore(tf.train.latest_checkpoint(model_path))

        
    
    @tf.function  #Training function
    def train_step_d(img):

        img = img + tf.random.normal([batch_size, 448, 448, 3], stddev=0.05) #make the discriminator more robust
        #noise = tf.random.normal([batch_size, 100]) #latent noise vector
        #alpha = tf.random.uniform([batch_size, 1, 1, 1], 0., 1.)
        
        #with tf.GradientTape() as grad:
            #generated_image = img_generator(noise, training=False)
            #interpolate = img + (alpha * (generated_image- img))
            #grad.watch(interpolate)
            #prediction = discriminator(interpolate, training=True)
        #gradient = grad.gradient(prediction, [interpolate])[0]
        #slopes = tf.sqrt(tf.reduce_sum(tf.square(gradient), axis = [1,2,3]))
        #gp_loss = tf.reduce_mean((slopes-1.)**2)

        with tf.GradientTape() as disc_tape:
            noise = tf.random.normal([batch_size, 100]) #latent noise vector
            generated_image = img_generator(noise, training=True)
            fake_output = discriminator(generated_image, training=True)
            real_output = discriminator(img, training=True)

            disc_loss = loss_utils.discriminator_loss_w(real_output, fake_output)
            gp_loss = loss_utils.gradient_penalty(generated_image, img, discriminator)
            total_loss = disc_loss + gp_lamda*gp_loss

        gradients_of_discriminator = disc_tape.gradient(total_loss, discriminator.trainable_variables)
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

        return disc_loss, gp_loss, real_output

    @tf.function  #Training function
    def train_step_g():
        noise = tf.random.normal([batch_size, 100]) #latent noise vector
        with tf.GradientTape() as gen_tape:
            generated_image = img_generator(noise, training=True)
            fake_output = discriminator(generated_image, training=True)

            gen_loss = loss_utils.generator_loss_w(fake_output)
    
        gradients_of_generator = gen_tape.gradient(gen_loss, img_generator.trainable_variables)
        generator_optimizer.apply_gradients(zip(gradients_of_generator, img_generator.trainable_variables))

        return gen_loss, fake_output, generated_image

    s= np.int64(0)
    for e in range(epochs):  #actual training

        for imagebatch in datareader.data:
            for _ in range(5):
                d_loss, gp_loss, real_output = train_step_d(imagebatch)
            g_loss, fake_output, fake_img = train_step_g()

            s += 1

            if s % 50 == 0:  #log every 50 training steps
                real_output = tf.reduce_mean(real_output)
                fake_output = tf.reduce_mean(fake_output)
                fake_img = tf.add(fake_img, float(1))
                fake_img = tf.multiply(fake_img, float(0.5))
                imagebatch = tf.add(imagebatch, float(1))
                imagebatch = tf.multiply(imagebatch, float(0.5))
                with writer.as_default():
                    summary.scalar("gen_loss", g_loss, description="train", step=s)
                    summary.scalar("disc_loss", d_loss, description="train", step=s)
                    summary.scalar("gp_loss", gp_loss, description="train", step=s)
                    summary.scalar("fake_output", fake_output, description="train", step=s)
                    summary.scalar("real_output", real_output, description="train", step=s)
                    summary.image("real_image", utils.cast_im(imagebatch), max_outputs=2, step=s)
                    summary.image("fake_image", utils.cast_im(fake_img), max_outputs=4, step=s)
                    print("Epoch: {}  Step: {}  Gen_Loss: {}  Disc_Loss: {}".format(e, s, g_loss, d_loss))
                    writer.flush()

            if s % 1000 == 0:
                checkpoint.save(file_prefix = checkpoint_prefix)

        


=======
import art_datareader
import matplotlib.pyplot as plt
import numpy as np
import generator
import loss_utils
import utils
import os
import sys

import tensorflow as tf
from tensorboard.plugins.beholder import Beholder
import tensorflow.summary as summary

#resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='grpc://' + os.environ['COLAB_TPU_ADDR'])
#tf.config.experimental_connect_to_cluster(resolver)
# This is the TPU initialization code that has to be at the beginning.
#tf.tpu.experimental.initialize_tpu_system(resolver)
#print("All devices: ", tf.config.list_logical_devices('TPU'))


epochs = int(100)
batch_size = 16
lr = 1e-4
gp_lamda = 10.
#dir = "C:\\Users\\samed\\OneDrive\\Dokumente\\Datasets\\ArtDataset\\TFrecord_train"
#logs_path = "C:\\Users\\samed\\OneDrive\\Dokumente\\Datasets\\Logs" 

directory = "/content/drive/My Drive/Art_Net/Dataset"
logs_path= "/content/drive/My Drive/Art_Net/Logs"
model_path ="/content/drive/My Drive/Art_Net/Model"

if __name__ == '__main__':
    #session_name = utils.get_session_name()
    session_logs_path = logs_path + "/" + "Test3"

    datareader = art_datareader.DataReader(batch_size, directory)
    img_generator = generator.Generator()
    discriminator = generator.Discriminator()

    #generator_optimizer = tf.keras.optimizers.RMSprop(learning_rate=lr)  #Wgan uses RMSprop
    #discriminator_optimizer = tf.keras.optimizers.RMSprop(learning_rate=lr) 

    generator_optimizer = tf.keras.optimizers.Adam(learning_rate=lr, beta_1 = 0.5, beta_2=0.9)  #Wgan gp uses adam
    discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=lr, beta_1 = 0.5, beta_2=0.9)

    writer = tf.summary.create_file_writer(session_logs_path, max_queue=0)
    checkpoint_prefix = os.path.join(model_path, "ckpt")
    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer, discriminator_optimizer=discriminator_optimizer, img_generator=img_generator, discriminator=discriminator)
        
    
    @tf.function  #Training function
    def train_step_d(img):

        img = img + tf.random.normal([batch_size, 448, 448, 3], stddev=0.05) #make the discriminator more robust
        noise = tf.random.normal([batch_size, 100]) #latent noise vector
        alpha = tf.random.uniform([batch_size, 1, 1, 1], 0., 1.)
        generated_image = img_generator(noise, training=True)
        interpolate = img + (alpha * (generated_image- img))
        with tf.GradientTape() as grad:
            grad.watch(interpolate)
            prediction = discriminator(interpolate, training=True)
        gradient = grad.gradient(prediction, [interpolate])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradient), axis = [1,2,3]))
        gp_loss = tf.reduce_mean((slopes-1.)**2)

        with tf.GradientTape() as disc_tape:
            disc_tape.watch(generated_image)
            fake_output = discriminator(generated_image, training=True)
            real_output = discriminator(img, training=True)

            disc_loss = loss_utils.discriminator_loss_w(real_output, fake_output)
            #gp_loss = loss_utils.gradient_penalty(generated_image, img, discriminator)
            disc_loss += gp_loss*gp_lamda

        gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
        disc_optimize = discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

        return disc_loss

    @tf.function  #Training function
    def train_step_g():
        noise = tf.random.normal([batch_size, 100]) #latent noise vector
        with tf.GradientTape() as gen_tape:
            generated_image = img_generator(noise, training=True)
            fake_output = discriminator(generated_image, training=True)

            gen_loss = loss_utils.generator_loss_w(fake_output)
    
        gradients_of_generator = gen_tape.gradient(gen_loss, img_generator.trainable_variables)
        gen_optimize = generator_optimizer.apply_gradients(zip(gradients_of_generator, img_generator.trainable_variables))

        return gen_loss, generated_image

    s= np.int64(0)
    for e in range(epochs):  #actual training

        for imagebatch in datareader.data:
            for _ in range(5):
                d_loss = train_step_d(imagebatch)
            g_loss, fake_img = train_step_g()

            s += 1

            if s % 50 == 0:  #log every 50 training steps

                with writer.as_default():
                    summary.scalar("gen_loss", g_loss, description="train", step=s)
                    summary.scalar("disc_loss", d_loss, description="train", step=s)
                    summary.image("real_image", utils.cast_im(imagebatch), max_outputs=3, step=s)
                    summary.image("fake_image", utils.cast_im(fake_img), max_outputs=3, step=s)
                    print("Epoch: {}  Step: {}  Gen_Loss: {}  Disc_Loss: {}".format(e, s, g_loss, d_loss))
                    writer.flush()

            if s % 1000 == 0:
                checkpoint.save(file_prefix = checkpoint_prefix)

        


>>>>>>> cb74623bf62f8b9870e362de937ce970953c3eba
