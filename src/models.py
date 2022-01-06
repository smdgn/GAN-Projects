import tensorflow as tf
import tensorflow_addons 
from tensorflow.keras.layers import BatchNormalization, Conv2D, \
    Dense, UpSampling2D, GlobalMaxPool2D, Flatten, Embedding, Concatenate

from tensorflow_addons.layers import SpectralNormalization


from layers import BConv2D, LConv2D, SelfAttention, ResBlock, ConditionalBatchNorm, Projection, Convolution, DenselyConnected, TransposedConvolution
from utils import cast_img
import metrics
from pathlib import Path
import datetime
import math


class WGANGP:

    def __init__(self, z_dim, h_dim, w_dim, c_dim, channels, scale, epochs, batch_size, lr_g, lr_d,
                 lambda_gp=10, noise_stddev=0.005, n_disc=5, n_samples=10):
        # Dimensions and starting channels
        self.z_dim = z_dim
        self.h_dim = h_dim
        self.w_dim = w_dim
        self.c_dim = c_dim
        self.channels = channels
        self.scale = scale

        # Training Hyperparamters, # TODO: Eventuell auslagern in einer TrainerKlasse für alle GANS
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr_g = lr_g
        self.lr_d = lr_d
        self.lambda_gp = lambda_gp
        self.g_opt = tf.keras.optimizers.Adam(lr=lr_g, beta_1=0., beta_2=0.9)
        self.d_opt = tf.keras.optimizers.Adam(lr=lr_d, beta_1=0., beta_2=0.9)
        self.noise_stddev = noise_stddev
        self.n_disc = n_disc
        self.n_samples = n_samples

        # Models
        self.g = self.build_generator()
        self.d = self.build_discriminator()

        self.g.summary()
        self.d.summary()
    

    def build_generator(self):
        inputs = tf.keras.Input(shape=(self.z_dim,))
        scale = pow(2, self.scale)
        x = Dense(math.ceil(self.h_dim / scale)*math.ceil(self.w_dim/scale)*self.channels, activation='relu')(inputs)
        x = tf.reshape(x, [-1, math.ceil(self.h_dim/scale), math.ceil(self.w_dim/scale), self.channels])

        for i in range(self.scale):
            x = BConv2D(x, self.channels//pow(2, i))
            x = UpSampling2D(2, interpolation='nearest')(x)
        
        x = BConv2D(x, self.channels//pow(2, self.scale))
        outputs = Conv2D(self.c_dim, 3, activation='tanh', padding='same')(x)
        model = tf.keras.Model(inputs, outputs, name='Generator')
        return model

    def build_discriminator(self):
        inputs = tf.keras.Input(shape=(self.h_dim, self.w_dim, self.c_dim))
        x = LConv2D(inputs, self.channels//pow(2, self.scale))
        for i in range(1,self.scale+1):
            x = LConv2D(x, self.channels//pow(2, self.scale-i), strides=2)
            x = LConv2D(x, self.channels//pow(2, self.scale-i),strides=1)
        x = Flatten()(x)
        outputs = Dense(1)(x)
        model = tf.keras.Model(inputs, outputs, name='Discriminator')
        return model

    def gradient_penalty(self, discriminator, real, fake):
        b, *_ = real.shape
        alpha = tf.random.uniform([b, 1, 1, 1], 0., 1.)
        interpolate = real + (alpha * (fake - real))
        with tf.GradientTape() as grad:
            grad.watch(interpolate)
            prediction = discriminator(interpolate, training=True)
        gradient = grad.gradient(prediction, [interpolate])[0]
        norm = tf.sqrt(tf.reduce_sum(tf.square(gradient), axis=[1, 2, 3]))
        return tf.reduce_mean((norm-1.)**2)

    def discriminator_loss(self, real_output, fake_output): 
        return tf.reduce_mean(fake_output) - tf.reduce_mean(real_output)

    def generator_loss(self, fake_output):
        return -tf.reduce_mean(fake_output)

    @tf.function  # Training function
    def train_step_g(self):
        noise = tf.random.normal((self.batch_size, self.z_dim))  # latent noise vector
        with tf.GradientTape() as gen_tape:
            generated_image = self.g(noise, training=True)
            fake_logits = self.d(generated_image, training=True)
            gen_loss = self.generator_loss(fake_logits)
        gradients_of_generator = gen_tape.gradient(gen_loss, self.g.trainable_variables)
        self.g_opt.apply_gradients(zip(gradients_of_generator, self.g.trainable_variables))
        return gen_loss

    @tf.function
    def train_step_d(self, real_image):
        #real_image = real_image + tf.random.normal((self.batch_size, self.h_dim, self.w_dim, 3),
                                                   #stddev=self.noise_stddev)
        noise = tf.random.normal((self.batch_size, self.z_dim))
        with tf.GradientTape() as disc_tape:
            generated_image = self.g(noise, training=True)
            fake_logits = self.d(generated_image, training=True)
            real_logits = self.d(real_image, training=True)
            disc_loss = self.discriminator_loss(real_logits, fake_logits)
            gp_loss = self.gradient_penalty(self.d, real_image, generated_image)
            disc_loss += gp_loss * self.lambda_gp
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.d.trainable_variables)
        self.d_opt.apply_gradients(zip(gradients_of_discriminator, self.d.trainable_variables))
        return disc_loss, fake_logits, real_logits

    @tf.function
    def generate(self, z=None):
        if z is None:
            z = tf.random.normal((self.n_samples, self.z_dim))
        return self.g(z, training=False)

    def train(self, data, verbose=100, log_n=None):
        s = tf.constant([0], dtype=tf.int64)
        z = tf.random.normal((self.n_samples, self.z_dim))

        metric_d_fake = tf.keras.metrics.Mean()
        metric_d_real = tf.keras.metrics.Mean()

        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        model_path = "Models/" + current_time
        Path(model_path).mkdir(parents=True, exist_ok=True)
        writer = None

        if log_n is not None:
            log_path = "Logs/" + current_time
            Path(log_path).mkdir(parents=True, exist_ok=True)
            writer = tf.summary.create_file_writer(log_path)
            
        template = 'Epoch: {}  Step: {}  G_loss: {}  D_loss {}'
        for e in range(self.epochs):
            for batch, *_ in data:
                for _ in range(self.n_disc):
                    d_loss, fake_logits, real_logits = self.train_step_d(batch)
                    metric_d_fake(fake_logits)
                    metric_d_real(real_logits)
                g_loss = self.train_step_g()
               
                if s % verbose == 0:
                    print(template.format(e, s, g_loss, d_loss))

                if writer is not None and s % log_n == 0:
                    image = self.generate(z)
                    with writer.as_default():
                        tf.summary.scalar("Fake Output", metric_d_fake.result(), step=s)
                        tf.summary.scalar("Real Output", metric_d_real.result(), step=s)
                        tf.summary.scalar("G Loss", g_loss, step=s)
                        tf.summary.scalar("D Loss", d_loss, step=s)
                        tf.summary.image("Generated Image", cast_img(image), step=s, max_outputs=self.n_samples)
                metric_d_real.reset_states()
                metric_d_fake.reset_states()
                s += 1
            self.g.save(model_path + "/generator")
            self.d.save(model_path + "/discriminator")

        print("Training finished successfully!")
        

class CWGANGP:

    def __init__(self, z_dim, h_dim, w_dim, c_dim, channels, scale, n_classes, embedding_dim, epochs, batch_size, lr_g, lr_d,
                 lambda_gp=10, noise_stddev=0.005, n_disc=5, n_samples=10):
        # Dimensions and starting channels
        self.z_dim = z_dim
        self.h_dim = h_dim
        self.w_dim = w_dim
        self.c_dim = c_dim
        self.channels = channels
        self.scale = scale
        self.n_classes = n_classes
        self.embedding_dim = embedding_dim

        # Training Hyperparamters, # TODO: Eventuell auslagern in einer TrainerKlasse für alle GANS
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr_g = lr_g
        self.lr_d = lr_d
        self.lambda_gp = lambda_gp
        self.g_opt = tf.keras.optimizers.Adam(lr=lr_g, beta_1=0., beta_2=0.9)
        self.d_opt = tf.keras.optimizers.Adam(lr=lr_d, beta_1=0., beta_2=0.9)
        self.noise_stddev = noise_stddev
        self.n_disc = n_disc
        self.n_samples = n_samples

        # Models
        self.g = self.build_generator()
        self.d = self.build_discriminator()

        self.g.summary()
        self.d.summary()

    def build_generator(self):
        scale = pow(2, self.scale)
        input_z = tf.keras.Input(shape=(self.z_dim,))
        input_label = tf.keras.Input(shape=(1,))
        y = Embedding(self.n_classes, self.embedding_dim)(input_label)
        y = Dense(math.ceil(self.h_dim / scale)*math.ceil(self.w_dim/scale), activation='relu')(y)
        y = tf.reshape(y, [-1, math.ceil(self.h_dim/scale), math.ceil(self.w_dim/scale), 1])
        
        x = Dense(math.ceil(self.h_dim / scale)*math.ceil(self.w_dim/scale)*self.channels, activation='relu')(input_z)
        x = tf.reshape(x, [-1, math.ceil(self.h_dim/scale), math.ceil(self.w_dim/scale), self.channels])
        xy = Concatenate()([x,y])

        for i in range(self.scale):
            xy = BConv2D(xy, self.channels//pow(2, i))
            xy = UpSampling2D(2, interpolation='nearest')(xy)
        
        xy = BConv2D(xy, self.channels//pow(2, self.scale))
        outputs = Conv2D(self.c_dim, 3, activation='tanh', padding='same')(xy)
        model = tf.keras.Model(inputs= [input_z, input_label], outputs=outputs, name='Generator')
        return model

    def build_discriminator(self):
        input_image = tf.keras.Input(shape=(self.h_dim, self.w_dim, self.c_dim))
        input_label = tf.keras.Input(shape=(1,))
        
        y = Embedding(self.n_classes, self.embedding_dim)(input_label)
        y = Dense(self.h_dim*self.w_dim, activation='relu')(y)
        y = tf.reshape(y, [-1, self.h_dim, self.w_dim, 1])
        
        xy = Concatenate()([input_image, y])
        xy = LConv2D(xy, self.channels//pow(2, self.scale))
        for i in range(1,self.scale+1):
            xy = LConv2D(xy, self.channels//pow(2, self.scale-i), strides=2)
            xy = LConv2D(xy, self.channels//pow(2, self.scale-i),strides=1)
        xy = Flatten()(xy)
        outputs = Dense(1)(xy)
        model = tf.keras.Model(inputs=[input_image, input_label], outputs=outputs, name='Discriminator')
        return model

    def gradient_penalty(self, discriminator, real, fake, real_label):
        b, *_ = real.shape
        alpha = tf.random.uniform([b, 1, 1, 1], 0., 1.)
        interpolate = real + (alpha * (fake - real))
        with tf.GradientTape() as grad:
            grad.watch(interpolate)
            prediction = discriminator([interpolate, real_label], training=True)
        gradient = grad.gradient(prediction, [interpolate])[0]
        norm = tf.sqrt(tf.reduce_sum(tf.square(gradient), axis=[1, 2, 3]))
        return tf.reduce_mean((norm-1.)**2)

    def discriminator_loss(self, real_output, fake_output): 
        return tf.reduce_mean(fake_output) - tf.reduce_mean(real_output)

    def generator_loss(self, fake_output):
        return -tf.reduce_mean(fake_output)

    @tf.function  # Training function
    def train_step_g(self):
        noise = tf.random.normal((self.batch_size, self.z_dim))  # latent noise vector
        label = tf.random.uniform(shape=(self.batch_size, 1), minval=0, maxval=self.n_classes, dtype=tf.int64)
        with tf.GradientTape() as gen_tape:
            generated_image = self.g([noise, label], training=True)
            fake_logits = self.d([generated_image, label], training=True)
            gen_loss = self.generator_loss(fake_logits)
        gradients_of_generator = gen_tape.gradient(gen_loss, self.g.trainable_variables)
        self.g_opt.apply_gradients(zip(gradients_of_generator, self.g.trainable_variables))
        return gen_loss

    @tf.function
    def train_step_d(self, real_image, real_label):
        #real_image = real_image + tf.random.normal((self.batch_size, self.h_dim, self.w_dim, 3),
                                                   #stddev=self.noise_stddev)
        noise = tf.random.normal((self.batch_size, self.z_dim))
        with tf.GradientTape() as disc_tape:
            generated_image = self.g([noise, real_label],training=True)
            fake_logits = self.d([generated_image, real_label], training=True)
            real_logits = self.d([real_image, real_label], training=True)
            disc_loss = self.discriminator_loss(real_logits, fake_logits)
            gp_loss = self.gradient_penalty(self.d, real_image, generated_image, real_label)
            disc_loss += gp_loss * self.lambda_gp
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.d.trainable_variables)
        self.d_opt.apply_gradients(zip(gradients_of_discriminator, self.d.trainable_variables))
        return disc_loss, fake_logits, real_logits

    @tf.function
    def generate(self, z=None, y=None):
        if z is None:
            z = tf.random.normal((self.n_samples, self.z_dim))
        if y is None:
            y = tf.random.uniform(shape=(self.n_samples, 1), minval=0, maxval=self.n_classes, dtype=tf.int64)
        return self.g([z, y], training=False)

    def train(self, data, verbose=100, log_n=None):
        s = tf.constant([0], dtype=tf.int64)
        z = tf.random.normal((self.n_samples, self.z_dim))
        #y = tf.random.uniform(shape=(self.n_samples, 1), minval=0, maxval=self.n_classes, dtype=tf.int64)
        y = tf.reshape(tf.range(0, self.n_classes, delta=1, dtype=tf.int64), [self.n_samples, 1])

        metric_d_fake = tf.keras.metrics.Mean()
        metric_d_real = tf.keras.metrics.Mean()

        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        model_path = "Models/" + current_time
        Path(model_path).mkdir(parents=True, exist_ok=True)
        writer = None

        if log_n is not None:
            log_path = "Logs/" + current_time
            Path(log_path).mkdir(parents=True, exist_ok=True)
            writer = tf.summary.create_file_writer(log_path)
            
        template = 'Epoch: {}  Step: {}  G_loss: {}  D_loss {}'
        for e in range(self.epochs):
            for batch, label in data:
                for _ in range(self.n_disc):
                    d_loss, fake_logits, real_logits = self.train_step_d(batch, label)
                    metric_d_fake(fake_logits)
                    metric_d_real(real_logits)
                g_loss = self.train_step_g()
               
                if s % verbose == 0:
                    print(template.format(e, s, g_loss, d_loss))

                if writer is not None and s % log_n == 0:
                    image = self.generate(z, y)
                    with writer.as_default():
                        tf.summary.scalar("Fake Output", metric_d_fake.result(), step=s)
                        tf.summary.scalar("Real Output", metric_d_real.result(), step=s)
                        tf.summary.scalar("G Loss", g_loss, step=s)
                        tf.summary.scalar("D Loss", d_loss, step=s)
                        tf.summary.image("Generated Image", cast_img(image), step=s, max_outputs=self.n_samples)
                metric_d_real.reset_states()
                metric_d_fake.reset_states()
                s += 1
            self.g.save(model_path + "/generator")
            self.d.save(model_path + "/discriminator")

        print("Training finished successfully!")
        

class CSAWGANGP:

    def __init__(self, z_dim, h_dim, w_dim, c_dim, channels, scale, n_classes, embedding_dim, c_scale, epochs, batch_size, lr_g, lr_d,
                 lambda_gp=10, noise_stddev=0.005, n_disc=5, n_samples=10):
        # Dimensions and starting channels
        self.z_dim = z_dim
        self.h_dim = h_dim
        self.w_dim = w_dim
        self.c_dim = c_dim
        self.channels = channels
        self.scale = scale
        self.n_classes = n_classes
        self.embedding_dim = embedding_dim
        self.c_scale = c_scale

        # Training Hyperparamters, # TODO: Eventuell auslagern in einer TrainerKlasse für alle GANS
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr_g = lr_g
        self.lr_d = lr_d
        self.lambda_gp = lambda_gp
        self.g_opt = tf.keras.optimizers.Adam(lr=lr_g, beta_1=0., beta_2=0.999)
        self.d_opt = tf.keras.optimizers.Adam(lr=lr_d, beta_1=0., beta_2=0.999)
        self.noise_stddev = noise_stddev
        self.n_disc = n_disc
        self.n_samples = n_samples

        # Models
        self.g = self.build_generator()
        self.d = self.build_discriminator()

        self.g.summary()
        self.d.summary()
        
    #def cast_attention(x):
        #b, *_ = x.shape
        #x = tf.reshape(x, [b, self.h_dim, self.w_dim, -1])

    def build_generator(self):
        scale = pow(2, self.scale)
        input_z = tf.keras.Input(shape=(self.z_dim,))
        input_label = tf.keras.Input(shape=(1,))
        #y = Embedding(self.n_classes, self.embedding_dim)(input_label)
        #y = tf.reshape(y, [-1, self.embedding_dim])
        #y = DenselyConnected(math.ceil(self.h_dim / scale)*math.ceil(self.w_dim/scale), sn=True, activation='relu')(y)
        #y = tf.reshape(y, [-1, math.ceil(self.h_dim/scale), math.ceil(self.w_dim/scale), 1])
        input_label_int = tf.cast(input_label, tf.int32)
        
        
        x = DenselyConnected(math.ceil(self.h_dim / scale)*math.ceil(self.w_dim/scale)*self.channels, sn=True, activation='relu')(input_z)
        xy = tf.reshape(x, [-1, math.ceil(self.h_dim/scale), math.ceil(self.w_dim/scale), self.channels])
        xy = ResBlock(filters=self.channels, num_classes=self.n_classes, sn=True, alpha=0.0, padding='same', kernel_initializer='glorot_normal')([xy, input_label_int])
        #x = DenselyConnected(math.ceil(self.h_dim / scale)*math.ceil(self.w_dim/scale)*256, sn=True, activation='relu')(input_z)
        #xy = tf.reshape(x, [-1, math.ceil(self.h_dim/scale), math.ceil(self.w_dim/scale), 256])
        #xy = Concatenate()([x,y])

        #z = tf.reshape(y, [-1,math.ceil(self.h_dim/scale)*math.ceil(self.w_dim/scale)])
        
        output_vector = []

        for i in range(1,self.scale+1):
            res = ConditionalBatchNorm(self.n_classes)([xy, input_label_int])
            res = tf.keras.activations.relu(res)
            res = Convolution(self.c_dim, 3, sn=True, activation='tanh', padding='same')(res)
            output_vector.insert(0, res)

            
            xy = ResBlock(filters=self.channels//pow(2,i), scale='up', num_classes=self.n_classes, sn=True, alpha=0.0, padding='same', kernel_initializer='glorot_normal')([xy, input_label_int])
            #xy = SpectralNormalization(Conv2D((self.channels//2)//pow(2, i), 3, padding='same'))(xy)
            #xy = ConditionalBatchNorm()([xy, z])
            #xy = tf.keras.activations.relu(xy)
            #if i == self.scale//2:
                #xy = SelfAttention(xy, c_scale=self.c_scale, sn=False)
            #xy = BConv2D(xy, self.channels//pow(2, i))
            #xy = UpSampling2D(2, interpolation='nearest')(xy)
            #if i<self.scale:
                
            
        
        #xy = ConditionalBatchNorm(self.n_classes)([xy, input_label_int])
        xy = BatchNormalization()(xy)
        xy = tf.keras.activations.relu(xy)
        outputs = Convolution(self.c_dim, 3, sn=True, activation='tanh', padding='same')(xy)
        output_vector.insert(0, outputs)
        model = tf.keras.Model(inputs=[input_z, input_label], outputs=output_vector, name='Generator')
        return model

    def build_discriminator(self):
        
        input_vector = []
        output_vector = []
        for i in range(self.scale+1):
            input_vector.append(tf.keras.Input(shape=(self.h_dim//pow(2, i), self.w_dim//pow(2,i), self.c_dim)))
        
        #input_image = tf.keras.Input(shape=(self.h_dim, self.w_dim, self.c_dim))
        input_label = tf.keras.Input(shape=(1,))
        
        #y = Embedding(self.n_classes, self.embedding_dim)(input_label)
        #y = SpectralNormalization(Dense(self.h_dim*self.w_dim, activation='relu'))(y)
        #y = tf.reshape(y, [-1, self.h_dim, self.w_dim, 1])
        
        #xy = Concatenate()([input_image, y])
        
        #xy = ResBlock(xy, self.channels//pow(2, self.scale-1))
        #xy = LConv2D(xy, self.channels//pow(2, self.scale))
        #xy = tf.keras.layers.LayerNormalization()(xy)
        #xy = tf.keras.activations.relu(xy, alpha=0.1)
        #xy = SpectralNormalization(Conv2D(self.channels//pow(2, self.scale), 3, padding='same'))(xy)
        #xy = tf.keras.activations.relu(xy, alpha=0.1)
        #xy = ResBlock(self.channels//pow(2, self.scale), scale='down', sn=True, alpha=0.1, padding='same')(input_image)
        #xy = ResBlock(self.channels//pow(2, self.scale), sn=True, alpha=0.1, padding='same')(xy)
        
        xy = input_vector[0]
        for i in range(0,self.scale):
            #xy = ResBlockDown(xy, self.channels//pow(2, self.scale-i))
            #xy = LConv2D(xy, self.channels//pow(2, self.scale-i), strides=2)
            #xy = LConv2D(xy, self.channels//pow(2, self.scale-i),strides=1)
            #xy = tf.keras.layers.LayerNormalization()(xy)
            #xy = tf.keras.activations.relu(xy, alpha=0.1)
            #xy = SpectralNormalization(Conv2D(self.channels//pow(2, self.scale-i), 3, strides=2, padding='same'))(xy)
            #xy = Conv2D(self.channels//pow(2, self.scale-i), 3, strides=1, padding='same')(xy)
            #xy = LConv2D(xy, self.channels//pow(2, self.scale-i),strides=1)
            xy = ResBlock(self.channels//pow(2, self.scale-i-1), scale='down', sn=True, alpha=0.1, padding='same')(xy)
            #xy = ResBlock(self.channels//pow(2, self.scale-i-1), sn=True, alpha=0.1, padding='same')(xy)
            #if i == self.scale//2:
                #xy = SelfAttention(xy, c_scale=self.c_scale, sn=False)
        #xy = Flatten()(xy)
        #outputs = SpectralNormalization(Dense(1))(xy)
        xy = ResBlock(self.channels, sn=True, alpha=0.1, padding='same')(xy)
        outputs = tf.keras.activations.relu(xy, alpha=0.1)
        outputs = Projection(outputs, input_label, self.n_classes, sn=True)
        output_vector.append(outputs)
        for i in range(1,self.scale+1):
            res_1 = tf.keras.activations.relu(input_vector[i], alpha=0.1)
            res_1 = Convolution(self.channels//2, 3, sn=True, padding='same')(res_1)
            res_1 = Projection(res_1, input_label, self.n_classes, sn=True)
            output_vector.append(res_1)
                                                                              
        input_vector.append(input_label)
        model = tf.keras.Model(inputs=input_vector, outputs=output_vector, name='Discriminator')
        
        return model

    def gradient_penalty(self, discriminator, real, fake, real_label):
        b, *_ = real.shape
        alpha = tf.random.uniform([b, 1, 1, 1], 0., 1.)
        interpolate = real + (alpha * (fake - real))
        with tf.GradientTape() as grad:
            grad.watch(interpolate)
            prediction = discriminator([interpolate, real_label], training=True)
        gradient = grad.gradient(prediction, [interpolate])[0]
        norm = tf.sqrt(tf.reduce_sum(tf.square(gradient), axis=[1, 2, 3]))
        return tf.reduce_mean((norm-1.)**2)

    def discriminator_loss(self, real_output_list, fake_output_list): 
        #return tf.reduce_mean(fake_output) - tf.reduce_mean(real_output)
        adv_loss = [tf.reduce_mean(tf.nn.relu(1.0 - real_output)) + tf.reduce_mean(tf.nn.relu(1.0 + fake_output)) for real_output, fake_output in zip(real_output_list, fake_output_list)]
        
        return sum(adv_loss)/len(adv_loss)
        

    def generator_loss(self, fake_output_list):
        loss = [-tf.reduce_mean(fake_output) for fake_output in fake_output_list]
        return sum(loss)/len(loss)

    @tf.function  # Training function
    def train_step_g(self):
        noise = tf.random.normal((self.batch_size, self.z_dim))  # latent noise vector
        label = tf.random.uniform(shape=(self.batch_size, 1), minval=0, maxval=self.n_classes, dtype=tf.int64)
        with tf.GradientTape() as gen_tape:
            generated_images = self.g([noise, label], training=True)
            fake_logits = self.d([*generated_images, label], training=True)
            gen_loss = self.generator_loss(fake_logits)
        gradients_of_generator = gen_tape.gradient(gen_loss, self.g.trainable_variables)
        self.g_opt.apply_gradients(zip(gradients_of_generator, self.g.trainable_variables))
        return gen_loss

    @tf.function
    def train_step_d(self, real_images, real_label):
        #real_image = real_image + tf.random.normal((self.batch_size, self.h_dim, self.w_dim, 3),
                                                   #stddev=self.noise_stddev)
        noise = tf.random.normal((self.batch_size, self.z_dim))
        label = tf.random.uniform(shape=(self.batch_size, 1), minval=0, maxval=self.n_classes, dtype=tf.int64)
        with tf.GradientTape() as disc_tape:
            generated_images = self.g([noise, label],training=True)
            fake_logits = self.d([*generated_images, label], training=True)
            real_logits = self.d([*real_images, real_label], training=True)
            disc_loss = self.discriminator_loss(real_logits, fake_logits)
            #gp_loss = self.gradient_penalty(self.d, real_image, generated_image, real_label)
            #disc_loss += gp_loss * self.lambda_gp
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.d.trainable_variables)
        self.d_opt.apply_gradients(zip(gradients_of_discriminator, self.d.trainable_variables))
        return disc_loss, fake_logits, real_logits

    @tf.function
    def generate(self, z=None, y=None):
        if z is None:
            z = tf.random.normal((self.n_samples, self.z_dim))
        if y is None:
            y = tf.random.uniform(shape=(self.n_samples, 1), minval=0, maxval=self.n_classes, dtype=tf.int64)
        image, *_ = self.g([z,y], training=False)  
        #attention_map = tf.reshape(tf.reduce_mean(attention, axis=-1), [self.n_samples, self.h_dim//2, self.w_dim//2, 1])
        #attention_map = tf.image.resize(attention_map, [self.h_dim, self.w_dim])
        
        return image

    def train(self, data, verbose=100, log_n=None):
        s = tf.constant([0], dtype=tf.int64)
        z = tf.random.normal((self.n_samples, self.z_dim))
        #y = tf.random.uniform(shape=(self.n_samples, 1), minval=0, maxval=self.n_classes, dtype=tf.int64)
        y = tf.reshape(tf.range(0, self.n_samples, delta=1, dtype=tf.int64), [self.n_samples, 1])

        metric_d_fake = tf.keras.metrics.Mean()
        metric_d_real = tf.keras.metrics.Mean()

        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        model_path = "Models/" + current_time
        Path(model_path).mkdir(parents=True, exist_ok=True)
        writer = None

        if log_n is not None:
            log_path = "Logs/" + current_time
            Path(log_path).mkdir(parents=True, exist_ok=True)
            writer = tf.summary.create_file_writer(log_path)
            
        template = 'Epoch: {}  Step: {}  G_loss: {}  D_loss {}'
        for e in range(self.epochs):
            for *batch, label in data:
                for _ in range(self.n_disc):
                    d_loss, fake_logits, real_logits = self.train_step_d(batch, label)
                    metric_d_fake(fake_logits)
                    metric_d_real(real_logits)
                g_loss = self.train_step_g()
               
                if s % verbose == 0:
                    print(template.format(e, s, g_loss, d_loss))

                if writer is not None and s % log_n == 0:
                    image = self.generate(z, y)
                    with writer.as_default():
                        tf.summary.scalar("Fake Output", metric_d_fake.result(), step=s)
                        tf.summary.scalar("Real Output", metric_d_real.result(), step=s)
                        tf.summary.scalar("G Loss", g_loss, step=s)
                        tf.summary.scalar("D Loss", d_loss, step=s)
                        tf.summary.image("Generated Image", cast_img(image), step=s, max_outputs=self.n_samples)
                        #tf.summary.image("Attention Map", attention_map, step=s, max_outputs=self.n_samples)
                metric_d_real.reset_states()
                metric_d_fake.reset_states()
                s += 1
            if e % 10 == 0:
                self.g.save(model_path + "/generator")
                self.d.save(model_path + "/discriminator")

        print("Training finished successfully!")
        
class ProjSNGAN:

    def __init__(self, z_dim, h_dim, w_dim, c_dim, channels, scale, n_classes, embedding_dim, c_scale, epochs, batch_size, lr_g, lr_d,
                 lambda_gp=10, noise_stddev=0.005, n_disc=5, n_samples=10):
        # Dimensions and starting channels
        self.z_dim = z_dim
        self.h_dim = h_dim
        self.w_dim = w_dim
        self.c_dim = c_dim
        self.channels = channels
        self.scale = scale
        self.n_classes = n_classes
        self.embedding_dim = embedding_dim
        self.c_scale = c_scale

        # Training Hyperparamters, # TODO: Eventuell auslagern in einer TrainerKlasse für alle GANS
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr_g = lr_g
        self.lr_d = lr_d
        self.lambda_gp = lambda_gp
        self.g_opt = tf.keras.optimizers.Adam(lr=lr_g, beta_1=0., beta_2=0.999)
        self.d_opt = tf.keras.optimizers.Adam(lr=lr_d, beta_1=0., beta_2=0.999)
        self.noise_stddev = noise_stddev
        self.n_disc = n_disc
        self.n_samples = n_samples

        # Models
        self.g = self.build_generator()
        self.d = self.build_discriminator()

        self.g.summary()
        self.d.summary()
        
    #def cast_attention(x):
        #b, *_ = x.shape
        #x = tf.reshape(x, [b, self.h_dim, self.w_dim, -1])

    def build_generator(self):
        scale = pow(2, self.scale)
        input_z = tf.keras.Input(shape=(self.z_dim,))
        input_label = tf.keras.Input(shape=(1,))
        #y = Embedding(self.n_classes, self.embedding_dim)(input_label)
        #y = tf.reshape(y, [-1, self.embedding_dim])
        #y = DenselyConnected(math.ceil(self.h_dim / scale)*math.ceil(self.w_dim/scale), sn=True, activation='relu')(y)
        #y = tf.reshape(y, [-1, math.ceil(self.h_dim/scale), math.ceil(self.w_dim/scale), 1])
        input_label_int = tf.cast(input_label, tf.int32)
        
        
        x = DenselyConnected(math.ceil(self.h_dim / scale)*math.ceil(self.w_dim/scale)*self.channels, sn=True, activation='relu')(input_z)
        xy = tf.reshape(x, [-1, math.ceil(self.h_dim/scale), math.ceil(self.w_dim/scale), self.channels])
        #xy = ResBlock(filters=self.channels, num_classes=self.n_classes, sn=True, alpha=0.0, padding='same', kernel_initializer='glorot_normal')([xy, input_label_int])
        #xy = TransposedConvolution(self.channels, 5, sn=True, padding='same', kernel_initializer='glorot_normal', strides=2)(xy)
        #xy = ConditionalBatchNorm(num_classes=self.n_classes)([xy, input_label_int])
        #xy = tf.keras.activations.relu(xy)
        #x = DenselyConnected(math.ceil(self.h_dim / scale)*math.ceil(self.w_dim/scale)*256, sn=True, activation='relu')(input_z)
        #xy = tf.reshape(x, [-1, math.ceil(self.h_dim/scale), math.ceil(self.w_dim/scale), 256])
        #xy = Concatenate()([x,y])

        #z = tf.reshape(y, [-1,math.ceil(self.h_dim/scale)*math.ceil(self.w_dim/scale)])

        for i in range(1,self.scale+1):    
            #xy = ResBlock(filters=self.channels//pow(2,i), scale='up', num_classes=self.n_classes, sn=True, alpha=0.0, padding='same', kernel_initializer='glorot_normal')([xy, input_label_int])
            xy = TransposedConvolution(self.channels//pow(2,i), 5, sn=True, padding='same', kernel_initializer='glorot_normal', strides=2)(xy)
           
            
            #xy = SpectralNormalization(Conv2D((self.channels//2)//pow(2, i), 3, padding='same'))(xy)
            xy = ConditionalBatchNorm(num_classes=self.n_classes)([xy, input_label_int])
            xy = tf.keras.activations.relu(xy)
            if i == 3:
                xy = SelfAttention(xy, c_scale=self.c_scale, sn=True)
            #xy = BConv2D(xy, self.channels//pow(2, i))
            #xy = UpSampling2D(2, interpolation='nearest')(xy)
            #if i<self.scale:
                
            
        
        #xy = ConditionalBatchNorm(self.n_classes)([xy, input_label_int])
        #xy = BatchNormalization()(xy)
        #xy = tf.keras.activations.relu(xy)
        outputs = Convolution(self.c_dim, 3, sn=True, activation='tanh', padding='same')(xy)
        model = tf.keras.Model(inputs=[input_z, input_label], outputs=outputs, name='Generator')
        return model

    def build_discriminator(self):
        
        input_image = tf.keras.Input(shape=(self.h_dim, self.w_dim, self.c_dim))
        input_label = tf.keras.Input(shape=(1,))
        
        #y = Embedding(self.n_classes, self.embedding_dim)(input_label)
        #y = SpectralNormalization(Dense(self.h_dim*self.w_dim, activation='relu'))(y)
        #y = tf.reshape(y, [-1, self.h_dim, self.w_dim, 1])
        
        #xy = Concatenate()([input_image, y])
        
        #xy = ResBlock(xy, self.channels//pow(2, self.scale-1))
        #xy = LConv2D(xy, self.channels//pow(2, self.scale))
        #xy = tf.keras.layers.LayerNormalization()(xy)
        #xy = tf.keras.activations.relu(xy, alpha=0.1)
        #xy = SpectralNormalization(Conv2D(self.channels//pow(2, self.scale), 3, padding='same'))(xy)
        #xy = tf.keras.activations.relu(xy, alpha=0.1)
        #xy = ResBlock(self.channels//pow(2, self.scale), scale='down', sn=True, alpha=0.1, padding='same')(input_image)
        #xy = ResBlock(self.channels//pow(2, self.scale), sn=True, alpha=0.1, padding='same')(xy)
        
        xy = input_image
        for i in range(0,self.scale):
            #xy = ResBlockDown(xy, self.channels//pow(2, self.scale-i))
            #xy = LConv2D(xy, self.channels//pow(2, self.scale-i), strides=2)
            #xy = LConv2D(xy, self.channels//pow(2, self.scale-i),strides=1)
            #xy = tf.keras.layers.LayerNormalization()(xy)
            #xy = tf.keras.activations.relu(xy, alpha=0.1)
            #xy = SpectralNormalization(Conv2D(self.channels//pow(2, self.scale-i), 3, strides=2, padding='same'))(xy)
            #xy = Conv2D(self.channels//pow(2, self.scale-i), 3, strides=1, padding='same')(xy)
            #xy = LConv2D(xy, self.channels//pow(2, self.scale-i),strides=1)
            
            #xy = ResBlock(self.channels//pow(2, self.scale-i-1), scale='down', sn=True, alpha=0.1, padding='same')(xy)
            xy = Convolution(self.channels//pow(2,self.scale-i-1), 5, sn=True, padding='same', strides=2)(xy)
            xy = tf.keras.activations.relu(xy, alpha=0.1)
            #xy = ResBlock(self.channels//pow(2, self.scale-i-1), sn=True, alpha=0.1, padding='same')(xy)
            #xy = ResBlock(self.channels//pow(2, self.scale-i-1), sn=True, alpha=0.1, padding='same')(xy)
            if i == 0:
                xy = SelfAttention(xy, c_scale=self.c_scale, sn=True)
        #xy = Flatten()(xy)
        #outputs = SpectralNormalization(Dense(1))(xy)
        #xy = ResBlock(self.channels, sn=True, alpha=0.1, padding='same')(xy)
        #xy = Convolution(self.channels, 3, sn=True, padding='same')(xy)
        outputs = tf.keras.activations.relu(xy, alpha=0.1)
        outputs = Projection(outputs, input_label, self.n_classes, sn=True)
        model = tf.keras.Model(inputs=[input_image,input_label], outputs=outputs, name='Discriminator')
        
        return model

    def gradient_penalty(self, discriminator, real, fake, real_label):
        b, *_ = real.shape
        alpha = tf.random.uniform([b, 1, 1, 1], 0., 1.)
        interpolate = real + (alpha * (fake - real))
        with tf.GradientTape() as grad:
            grad.watch(interpolate)
            prediction = discriminator([interpolate, real_label], training=True)
        gradient = grad.gradient(prediction, [interpolate])[0]
        norm = tf.sqrt(tf.reduce_sum(tf.square(gradient), axis=[1, 2, 3]))
        return tf.reduce_mean((norm-1.)**2)

    def discriminator_loss(self, real_output, fake_output): 
        #return tf.reduce_mean(fake_output) - tf.reduce_mean(real_output)
        return tf.reduce_mean(tf.nn.relu(1.0 - real_output)) + tf.reduce_mean(tf.nn.relu(1.0 + fake_output))
        

    def generator_loss(self, fake_output):
        return -tf.reduce_mean(fake_output)

    @tf.function  # Training function
    def train_step_g(self):
        noise = tf.random.normal((self.batch_size, self.z_dim))  # latent noise vector
        label = tf.random.uniform(shape=(self.batch_size, 1), minval=0, maxval=self.n_classes, dtype=tf.int64)
        with tf.GradientTape() as gen_tape:
            
            generated_images = self.g([noise, label], training=True)
            fake_logits = self.d([generated_images, label], training=True)
            gen_loss = self.generator_loss(fake_logits)
        gradients_of_generator = gen_tape.gradient(gen_loss, self.g.trainable_variables)
        self.g_opt.apply_gradients(zip(gradients_of_generator, self.g.trainable_variables))
        gradient_norm = tf.linalg.global_norm(gradients_of_generator)
        return gen_loss, gradient_norm

    @tf.function
    def train_step_d2(self, real_images, real_label):
        #real_image = real_image + tf.random.normal((self.batch_size, self.h_dim, self.w_dim, 3),
                                                   #stddev=self.noise_stddev)
        noise = tf.random.normal((self.batch_size, self.z_dim))
        label = tf.random.uniform(shape=(self.batch_size, 1), minval=0, maxval=self.n_classes, dtype=tf.int64)
        with tf.GradientTape() as disc_tape:
            
            generated_images = self.g([noise, label],training=True)
            fake_logits = self.d([generated_images, label], training=True)
            real_logits = self.d([real_images, real_label], training=True)
            disc_loss = self.discriminator_loss(real_logits, fake_logits)
            #gp_loss = self.gradient_penalty(self.d, real_image, generated_image, real_label)
            #disc_loss += gp_loss * self.lambda_gp
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.d.trainable_variables)
        self.d_opt.apply_gradients(zip(gradients_of_discriminator, self.d.trainable_variables))
        gradient_norm = tf.linalg.global_norm(gradients_of_discriminator)
        return disc_loss, fake_logits, real_logits, gradient_norm
    
    @tf.function
    def train_step_d(self, real_images, real_label):
        real_image = real_image + tf.random.normal((self.batch_size, self.h_dim, self.w_dim, 3),
                                                   stddev=self.noise_stddev)
        noise = tf.random.normal((self.batch_size, self.z_dim))
        label = tf.random.uniform(shape=(self.batch_size, 1), minval=0, maxval=self.n_classes, dtype=tf.int64)
        with tf.GradientTape() as disc_tape, tf.GradientTape() as gen_tape:
            
            generated_images = self.g([noise, real_label],training=True)
            fake_logits = self.d([generated_images, real_label], training=True)
            real_logits = self.d([real_images, real_label], training=True)
            disc_loss = self.discriminator_loss(real_logits, fake_logits)
            gen_loss = self.generator_loss(fake_logits)
            #gp_loss = self.gradient_penalty(self.d, real_image, generated_image, real_label)
            #disc_loss += gp_loss * self.lambda_gp
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.d.trainable_variables)
        gradients_of_generator = gen_tape.gradient(gen_loss, self.g.trainable_variables)
        self.d_opt.apply_gradients(zip(gradients_of_discriminator, self.d.trainable_variables))
        self.g_opt.apply_gradients(zip(gradients_of_generator, self.g.trainable_variables))
        return disc_loss, fake_logits, real_logits, gen_loss

    @tf.function
    def generate(self, z=None, y=None, n_samples=None):
        if n_samples is None:
            n_samples = self.n_samples
        if z is None:
            z = tf.random.normal((n_samples, self.z_dim))
        if y is None:
            y = tf.random.uniform(shape=(n_samples, 1), minval=0, maxval=self.n_classes, dtype=tf.int64)
        image = self.g([z,y], training=False)  
        #attention_map = tf.reshape(tf.reduce_mean(attention, axis=-1), [self.n_samples, self.h_dim//2, self.w_dim//2, 1])
        #attention_map = tf.image.resize(attention_map, [self.h_dim, self.w_dim])
        
        return image

    def train(self, data, verbose=100, log_n=None):
        s = tf.constant([0], dtype=tf.int64)
        z = tf.random.normal((self.n_samples, self.z_dim))
        #y = tf.random.uniform(shape=(self.n_samples, 1), minval=0, maxval=self.n_classes, dtype=tf.int64)
        y = tf.reshape(tf.range(0, self.n_samples, delta=1, dtype=tf.int64), [self.n_samples, 1])
        
        y_list = [tf.fill([self.n_samples, 1], i) for i in range(self.n_classes)]

        metric_d_fake = tf.keras.metrics.Mean()
        metric_d_real = tf.keras.metrics.Mean()
        #fid = metrics.IntraFID(input_shape=(75,75,3), num_samples = self.n_samples)

        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        model_path = "Models/" + current_time
        Path(model_path).mkdir(parents=True, exist_ok=True)
        writer = None
        
        
        if log_n is not None:
            log_path = "Logs/" + current_time
            Path(log_path).mkdir(parents=True, exist_ok=True)
            writer = tf.summary.create_file_writer(log_path)
            
        template = 'Epoch: {}  Step: {}  G_loss: {}  D_loss {}'
        for e in range(self.epochs):
            for batch, label in data:
                
                
                for _ in range(self.n_disc):
                    d_loss, fake_logits, real_logits, d_norm = self.train_step_d2(batch, label)
                metric_d_fake(fake_logits)
                metric_d_real(real_logits)
                g_loss, g_norm = self.train_step_g()
               
                if s % verbose == 0:
                    print(template.format(e, s, g_loss, d_loss))

                if writer is not None and s % log_n == 0:
                    image = self.generate(z, y)
                    #fake_images = [self.generate(z,y2) for y2 in y_list]
                    #fid_classes, fid_scalar = fid.get_fid(fake_images)
                    
                    with writer.as_default():
                        #tf.summary.scalar("/1 Fake Output", metric_d_fake.result(), step=s)
                        #tf.summary.scalar("/1 Real Output", metric_d_real.result(), step=s)
                        tf.summary.scalar("/0 Discriminator Scale", self.d.get_layer("scale_1").get_weights()[0], step=s)
                        tf.summary.scalar("/0 Generator Scale", self.g.get_layer("scale").get_weights()[0], step=s)
                        tf.summary.scalar("/1 Fake Output", tf.reduce_mean(fake_logits), step=s)
                        tf.summary.scalar("/1 Real Output", tf.reduce_mean(real_logits), step=s)
                        tf.summary.scalar("/2 G Loss", g_loss, step=s)
                        tf.summary.scalar("/2 D Loss", d_loss, step=s)
                        tf.summary.scalar("/3 D Gradient Norm", d_norm, step=s)
                        tf.summary.scalar("/3 G Gradient Norm", g_norm, step=s)
                        tf.summary.image("Generated Image", cast_img(image), step=s, max_outputs=self.n_samples)
                        #tf.summary.image("Attention Map", attention_map, step=s, max_outputs=self.n_samples) 
                        #tf.summary.scalar("/4 Mean FID", fid_scalar, step=e)
                
                #metric_d_real.reset_states()
                #metric_d_fake.reset_states()
                s += 1
            if e % 10 == 0 and e != 0:
                
                #fake_images = [self.generate(z,y2) for y2 in y_list]
                #fid_classes, fid_scalar = fid.get_fid(fake_images)
                ssim_list = metrics.sample_ssim(self.g, self.n_classes, self.z_dim, n_samples=100)
                with writer.as_default():
                        for count, ssim_class in enumerate(ssim_list):
                            tf.summary.scalar("/4 MS-SSIM for Class {}".format(count), ssim_class, step=s)
                        #tf.summary.scalar("/5 Mean FID", fid_scalar, step=e)
                        #for count, fid_class in enumerate(fid_classes):
                            #tf.summary.scalar("/5 Class FID {}".format(count), fid_class, step=e)
                self.g.save(model_path + "/generator")
                self.d.save(model_path + "/discriminator")
           # if e == 10:
                #self.n_disc = 2
            #if e == 100:
                #self.n_disc = 3

        print("Training finished successfully!")



