import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import BatchNormalization, Conv2D, Conv2DTranspose, Dense, LayerNormalization, UpSampling2D
from tensorflow.keras.activations import relu

class BConv2DTranspose(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.conv_transpose = Conv2DTranspose(*args, **kwargs)
        self.batchnorm = BatchNormalization()
    
    def call(self, x):
        x = self.conv_transpose(x)
        x = self.batchnorm(x)
        return relu(x, alpha=0.3)

class BConv2D(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.conv = Conv2D(*args, **kwargs)
        self.batch_norm = BatchNormalization()

    def call(self, x):
        x = self.conv(x)
        x = self.batch_norm(x)
        return relu(x, alpha=0.3)

class LConv2D(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.conv = Conv2D(*args, **kwargs)
        self.layer_norm = LayerNormalization()

    def call(self, x):
        x = self.conv(x)
        x = self.layer_norm(x)
        return relu(x, alpha = 0.3)

class Generator(tf.keras.Model):
    def __init__(self):
        super().__init__()

        kwargs = {'strides': 2, 'padding': 'SAME'}

        self.d1 = Dense(14*14*512, input_shape=(100,), activation='relu')
        self.conv1 = BConv2D(512, 3, strides=1, padding='SAME')
        self.upscale1 = UpSampling2D(interpolation='bilinear')

        self.conv2 = BConv2D(256, 3, strides=1, padding='SAME')
        self.upscale2 = UpSampling2D(interpolation='bilinear')

        self.conv3 = BConv2D(128, 3, strides=1, padding='SAME')
        self.upscale3 = UpSampling2D(interpolation='bilinear')

        self.conv4 = BConv2D(64, 3, strides=1, padding='SAME')
        self.upscale4 = UpSampling2D(interpolation='bilinear')

        self.conv5 = BConv2D(32, 3, strides=1, padding='SAME')
        self.upscale5 = UpSampling2D(interpolation='bilinear')
        self.out = Conv2D(3, 3, activation='tanh', padding='SAME')

    def call(self, x):
        x1 = self.d1(x)
        x1 = tf.reshape(x1, [-1,14,14,512])
        
        x2 = self.conv1(x1)
        x2 = self.upscale1(x2)

        x2 = self.conv2(x2)
        x2 = self.upscale2(x2)

        x2 = self.conv3(x2)
        x2 = self.upscale3(x2)

        x2 = self.conv4(x2)
        x2 = self.upscale4(x2)

        x2 = self.conv5(x2)
        x2 = self.upscale5(x2)

        output = self.out(x2)
        return output

class Discriminator(tf.keras.Model):
    def __init__(self):
        super().__init__()

        kwargs = {'strides': 2, 'padding': 'same'}

        self.conv1 = LConv2D(32, 7, **kwargs)
        self.conv2 = LConv2D(64, 5, **kwargs)
        self.conv3 = LConv2D(128, 3,**kwargs)
        self.conv4 = LConv2D(256, 3,**kwargs)
        self.conv5 = LConv2D(512, 3,**kwargs)
        self.pool  = tf.keras.layers.GlobalMaxPooling2D()
        self.d1 = Dense(1)

    def call(self, x):
        x1 = self.conv1(x)
        b, w, h, c = x1.shape
        x = tf.image.resize(x, [w, h])
        x1 = tf.concat([x1, x], -1)    #residual block, depth concat

        x1 = self.conv2(x1)
        b, w, h, c = x1.shape
        x = tf.image.resize(x, [w, h])
        x1 = tf.concat([x1, x], -1)

        x1 = self.conv3(x1)
        b, w, h, c = x1.shape
        x = tf.image.resize(x, [w, h])
        x1 = tf.concat([x1, x], -1)

        x1 = self.conv4(x1)
        b, w, h, c = x1.shape
        x = tf.image.resize(x, [w, h])
        x1 = tf.concat([x1, x], -1)

        x1 = self.conv5(x1)

        x1 = self.pool(x1)
        b, *_ = x1.shape
        x1 = tf.reshape(x1, [b, -1])
        decision = self.d1(x1)
        return decision










