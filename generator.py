import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import BatchNormalization, Conv2D, Conv2DTranspose, Dense
from tensorflow.keras.activations import relu, elu 

class BConv2DTranspose(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.conv_transpose = Conv2DTranspose(*args, **kwargs)
        self.batchnorm = BatchNormalization()
    
    def call(self, x):
        x = self.conv_transpose(x)
        x = self.batchnorm(x)
        return relu(x, alpha = 0.3)

class BConv2D(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.conv = Conv2D(*args, **kwargs)
        self.batch_norm = BatchNormalization()

    def call(self, x):
        x = self.conv(x)
        x = self.batch_norm(x)
        return relu(x, alpha = 0.3)

class Generator(tf.keras.Model):
    def __init__(self):
        super().__init__()

        kwargs = {'strides': 2, 'padding': 'SAME'}

        self.d1 = Dense(14*14*512, input_shape=(100,), activation='relu')
        self.dconv1 = BConv2DTranspose(512, 3, **kwargs)
        self.dconv15 = BConv2D(512, 3, strides=1, padding='SAME')
        self.dconv2 = BConv2DTranspose(256, 3, **kwargs)
        self.dconv25 = BConv2D(256, 3, strides=1, padding='SAME')
        self.dconv3 = BConv2DTranspose(128, 3, **kwargs)
        self.dconv35 = BConv2D(128, 3, strides=1, padding='SAME')
        self.dconv4 = BConv2DTranspose(64, 5, **kwargs)
        self.dconv45 = BConv2D(64, 5, strides=1, padding='SAME')
        self.out = Conv2DTranspose(3, 5, activation='tanh', **kwargs)

    def call(self, x):
        x1 = self.d1(x)
        x1 = tf.reshape(x1, [-1,14,14,512])
        
        x2 = self.dconv1(x1)
        x2 = self.dconv15(x2)

        x2 = self.dconv2(x2)
        x2 = self.dconv25(x2)

        x2 = self.dconv3(x2)
        x2 = self.dconv35(x2)

        x2 = self.dconv4(x2)
        x2 = self.dconv45(x2)

        output = self.out(x2)
        return output

class Discriminator(tf.keras.Model):
    def __init__(self):
        super().__init__()

        kwargs = {'strides': 2, 'padding': 'SAME'}

        self.conv1 = BConv2D(32, 5, **kwargs)
        self.conv2 = BConv2D(64, 5, **kwargs)
        self.conv3 = BConv2D(128, 3, **kwargs)
        self.conv4 = BConv2D(256, 3, **kwargs)
        self.conv5 = BConv2D(512, 3, **kwargs)
        self.pool  = tf.keras.layers.GlobalAveragePooling2D()
        self.d1 = Dense(1, activation = "sigmoid")

    def call(self, x):
        x1 = self.conv1(x)
        x1 = self.conv2(x1)
        x1 = self.conv3(x1)
        x1 = self.conv4(x1)
        x1 = self.conv5(x1)
        x1 = self.pool(x1)
        b, *_ = x1.shape
        x1 = tf.reshape(x1, [b, -1])
        decision = self.d1(x1)
        return decision










