import art_datareader
import matplotlib.pyplot as plt
import numpy as np
import generator
import loss_utils
import utils
import os

import tensorflow as tf
from tensorboard.plugins.beholder import Beholder
import tensorflow.summary as summary


#tf.enable_eager_execution()

#dir = r"C:\Users\samed\OneDrive\Dokumente\Datasets\ArtDataset\TFrecord_train"
#datareader = art_datareader.DataReader(1, dir)

#with tf.Session() as sess:
    #tf.global_variables_initializer().run()
    #while(1):
        #frame = datareader.read()
        #frame = sess.run(frame)
        #imgplot = plt.imshow(np.squeeze(frame, axis=0))
        #plt.show()
S_max = int(50000)
batch_size = 16
lr = 1e-4

#dir = "C:\\Users\\samed\\OneDrive\\Dokumente\\Datasets\\ArtDataset\\TFrecord_train"
#logs_path = "C:\\Users\\samed\\OneDrive\\Dokumente\\Datasets\\Logs" 

directory = "/content/drive/My Drive/Art_Net/Dataset"
logs_path= "/content/drive/My Drive/Art_Net/Logs"

if __name__ == '__main__':
    #session_name = utils.get_session_name()
    session_logs_path = logs_path + "/" + "Test2"
    glob_step = tf.Variable(0, name='global_step', trainable=False) 

    datareader = art_datareader.DataReader(batch_size, directory)
    img_generator = generator.Generator()
    discriminator = generator.Discriminator()

    #generator_optimizer = tf.train.AdamOptimizer(learning_rate = lr)
    #discriminator_optimizer = tf.train.AdamOptimizer(learning_rate= lr)

    generator_optimizer = tf.keras.optimizers.Adam(lr)
    discriminator_optimizer = tf.keras.optimizers.Adam(lr)

    beholder = Beholder(session_logs_path)

    img = datareader.read()
        
    noise = tf.random.normal([batch_size, 100])
    img = img + tf.random.normal([16, 550, 600, 3], stddev=0.1)

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      generated_image = img_generator(noise, training=True)
      fake_output = discriminator(generated_image, training=True)
      real_output = discriminator(img, training=True)

      gen_loss = loss_utils.generator_loss(fake_output)
      disc_loss = loss_utils.discriminator_loss(real_output, fake_output)
    
    gradients_of_generator = gen_tape.gradient(gen_loss, img_generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    gen_optimize = generator_optimizer.apply_gradients(zip(gradients_of_generator, img_generator.trainable_variables))
    disc_optimize = discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    #gen_optimize = generator_optimizer.minimize(gen_loss+disc_loss, global_step=tf.train.get_global_step())
    #disc_optimize = discriminator_optimizer.minimize(disc_loss, global_step=tf.train.get_global_step())

    fake_metric = tf.reduce_sum(fake_output)/batch_size
    real_metric = tf.reduce_sum(real_output)/batch_size

    summary.scalar("gen_loss", gen_loss, family="train")
    summary.scalar("disc_loss", disc_loss, family="train")
    summary.scalar("fake_metric", fake_metric, family="train")
    summary.scalar("real_metric", real_metric, family="train")
    summary.image("real_image", utils.cast_im(img), max_outputs=3)
    summary.image("fake_image", utils.cast_im(generated_image), max_outputs=3)

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        writer = summary.FileWriter(session_logs_path, sess.graph, max_queue=0)

            
        for s in range(S_max):
            dl, *_ = sess.run([disc_loss, disc_optimize])
            gl, *_ = sess.run([gen_loss, gen_optimize])
            sess.run(gen_optimize)
            beholder.update(session=sess)

            if s % 50 == 0:
                writer.add_summary(sess.run(summary.merge_all()), s)
                fake, real = sess.run([fake_metric, real_metric])
                print("Iteration: {} Generator_Loss: {} Discriminator_Loss: {}".format(s, gl, dl))
                print("Fake Metrix: {} Real Metric: {}".format(fake, real))


