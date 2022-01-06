import tensorflow as tf
import tensorflow_addons 
from tensorflow.keras.layers import BatchNormalization, Conv2D, LayerNormalization, MaxPool2D, AveragePooling2D, UpSampling2D, Conv2DTranspose, Dense, Flatten, ReLU, LeakyReLU, Embedding
from tensorflow.keras.activations import relu
from tensorflow_addons.layers import SpectralNormalization


class Convolution(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, sn=False, *args, **kwargs):
        super(Convolution, self).__init__()
        if sn:
            self.conv = SpectralNormalization(Conv2D(filters, kernel_size, *args, **kwargs))
        else: 
            self.conv = Conv2D(filters, kernel_size, *args, **kwargs)
     
    def call(self, x):
        return self.conv(x)
    
    
class TransposedConvolution(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, sn=False, *args, **kwargs):
        super(TransposedConvolution, self).__init__()
        if sn:
            self.conv = SpectralNormalization(Conv2DTranspose(filters, kernel_size, *args, **kwargs))
        else: 
            self.conv = Conv2DTranspose(filters, kernel_size, *args, **kwargs)
     
    def call(self, x):
        return self.conv(x)



class DenselyConnected(tf.keras.layers.Layer):
    def __init__(self, units, sn=False, *args, **kwargs):
        super(DenselyConnected, self).__init__()
        if sn:
            self.dense = SpectralNormalization(Dense(units, *args, **kwargs))
        else: 
            self.dense = Dense(units, *args, **kwargs)
     
    def call(self, x):
        return self.dense(x)
    

class ResBlock(tf.keras.layers.Layer):
    def __init__(self, filters, scale=None, norm=None, num_classes=None, sn=False, alpha=None, **kwargs):
        super(ResBlock, self).__init__()
        self.block, self.skip = self._get_block(filters, scale, norm, num_classes, sn, alpha, **kwargs)
        
        self.num_classes = num_classes
     
    def call(self, inputs):
        
        if self.num_classes is not None:
            x, label = inputs
        else:
            x = inputs
        skip = x
        for layer in self.block:
            if isinstance(layer, ConditionalBatchNorm):
                x = layer([x, label])
            else:
                x = layer(x)
        if self.skip is not None:
            for layer in self.skip:
                skip = layer(skip)
        return skip + x
    
    def _get_block(self, filters, scale, norm, num_classes, sn, alpha, **kwargs):
        if scale is None:
            block = list(filter(None,[
                self._get_norm(norm, num_classes),
                self._get_activation(alpha), 
                Convolution(filters, 3, sn=sn, **kwargs), 
                self._get_norm(norm, num_classes),
                self._get_activation(alpha), 
                Convolution(filters, 3, sn=sn, **kwargs)]))
           
            skip = None
                                     
        elif scale.lower() == 'up':
            block = list(filter(None,[
                self._get_norm(norm, num_classes),
                self._get_activation(alpha), 
                UpSampling2D(2, interpolation='nearest'),
                Convolution(filters, 3, sn=sn, **kwargs), 
                self._get_norm(norm, num_classes),
                self._get_activation(alpha), 
                Convolution(filters, 3, sn=sn, **kwargs)]))
                                     
            skip = [
                UpSampling2D(2, interpolation='nearest'),
                Convolution(filters, 1, sn=sn, **kwargs)]
                                     
        elif scale.lower() == 'down':
            block = list(filter(None,[
                self._get_norm(norm, num_classes),
                self._get_activation(alpha), 
                Convolution(filters, 3, sn=sn, **kwargs), 
                self._get_norm(norm, num_classes),
                self._get_activation(alpha), 
                Convolution(filters, 3, sn=sn, **kwargs),
                AveragePooling2D(pool_size=2, strides=2)]))
            
            skip = [
                Convolution(filters, 1, sn=sn, **kwargs),
                AveragePooling2D(pool_size=2, strides=2)]
                                      
        return block, skip
            
           
    def _get_activation(self, alpha):
        if alpha is  None:
            return None
        elif alpha > 0.0: 
            return LeakyReLU(alpha=alpha)
        else:
            return ReLU()
        
            
    #def _get_scale(self, scale):
        #if scale is None: 
            #return None
        #elif scale.lower() == 'up':
            #return UpSampling2D(2, interpolation='nearest')
        #elif scale.lower() == 'down':
             #return AveragePooling2D(pool_size=2, strides=2)
        
    
    def _get_norm(self, norm, num_classes):
        if num_classes is not None:
            return ConditionalBatchNorm(num_classes)
        elif norm is None:
            return None
        elif norm.lower() == 'batch':
            return BatchNormalization()
        elif scale.lower() == 'layer':
             return LayerNormalization()
            
        
def Projection(inputs, labels, num_classes, sn=False):
    # Global pool input feature maps
    b, h, w, c = inputs.shape
    pooled = tf.reduce_sum(inputs, axis=(1,2)) #[B, C]
    
    # Get Output
    out = DenselyConnected(1, sn=sn)(pooled) # [B, 1]
    
    # Create Inner Product
    y = Embedding(num_classes, c)(labels) #[B, 1, C]
    y = tf.reshape(y, [-1, c])
    inner = tf.reduce_sum(pooled * y, axis=1) #[B, 1]
    out += inner 
    return out
      

class ConditionalBatchNorm(tf.keras.layers.Layer):
    
    
    def __init__(self, num_classes, momentum=0.9, epsilon=1e-5):
        super(ConditionalBatchNorm, self).__init__()
        self.momentum = momentum
        self.epsilon = epsilon
        self.num_classes = num_classes
        
        
    def build(self, input_shape):
        self.channels=input_shape[0][-1]
        self.beta = self.add_weight(name="beta", shape=(self.num_classes, self.channels), initializer="zeros", trainable=True)
        self.gamma = self.add_weight(name="gamma", shape=(self.num_classes, self.channels), initializer="ones", trainable=True)
        self.moving_mean = self.add_weight(name="moving_mean", shape=(self.channels),initializer="zeros",trainable=False, dtype=tf.float32)
        self.moving_var = self.add_weight(name = "moving_var", shape=(self.channels),initializer="ones",trainable=False, dtype=tf.float32)
            
    def call(self, inputs, training=True):
        batch, labels = inputs
      
        beta = tf.gather(self.beta, labels)
        gamma = tf.gather(self.gamma, labels)
        
        beta = tf.reshape(beta, shape=[-1, 1, 1, self.channels])
        gamma = tf.reshape(gamma, shape=[-1, 1, 1, self.channels]) 

        if training:
            batch_mean, batch_var = tf.nn.moments(batch, [0,1,2])
            self.moving_mean.assign(self.moving_mean * self.momentum + batch_mean * (1 - self.momentum))
            self.moving_var.assign(self.moving_var * self.momentum + batch_var * (1 - self.momentum))
            return tf.nn.batch_normalization(batch, batch_mean, batch_var, beta, gamma, self.epsilon)
        else:
            return tf.nn.batch_normalization(batch, self.moving_mean, self.moving_var, beta, gamma, self.epsilon)
        

class ConditionalDenseBatchNorm(tf.keras.layers.Layer):
    
    
    def __init__(self, momentum=0.9, epsilon=1e-5):
        super(ConditionalDenseBatchNorm, self).__init__()
        self.momentum = momentum
        self.epsilon = epsilon
                
    def build(self, input_shape):
        self.channels=input_shape[0][-1]
        self.beta = Dense(self.channels, trainable=True)
        self.gamma = Dense(self.channels, trainable=True)
        self.moving_mean = self.add_weight(name="moving_mean", shape=(self.channels),initializer="zeros",trainable=False, dtype=tf.float32)
        self.moving_var = self.add_weight(name = "moving_var", shape=(self.channels),initializer="ones",trainable=False, dtype=tf.float32)
            
    def call(self, inputs, training=True):
        batch, labels = inputs
      
        beta = self.beta(labels)
        gamma = self.gamma(labels)
        
        beta = tf.reshape(beta, shape=[-1, 1, 1, self.channels])
        gamma = tf.reshape(gamma, shape=[-1, 1, 1, self.channels]) + 1.0

        if training:
            batch_mean, batch_var = tf.nn.moments(batch, [0,1,2])
            self.moving_mean.assign(self.moving_mean * self.momentum + batch_mean * (1 - self.momentum))
            self.moving_var.assign(self.moving_var * self.momentum + batch_var * (1 - self.momentum))
            return tf.nn.batch_normalization(batch, batch_mean, batch_var, beta, gamma, self.epsilon)
        else:
            return tf.nn.batch_normalization(batch, self.moving_mean, self.moving_var, beta, gamma, self.epsilon)


class Scale(tf.keras.layers.Layer):
    def __init__(self, init=0.):
        super(Scale, self).__init__()
        self.scale = tf.Variable(init, name="scale", trainable=True)

    def call(self, inputs):
        return inputs * self.scale


def SelfAttention(inputs, c_scale=8, sn=False, **kwargs):
    b, h, w, c = inputs.shape
    b = tf.shape(inputs)[0]
    channels = c//c_scale  #=C
    assert channels > 0
    query = Convolution(channels, 1, sn=sn, padding='same', **kwargs)(inputs) #[b, h, w, C]
    #query = MaxPool2D(pool_size=2, strides=2, padding='same')(query)
    
    key = Convolution(channels, 1, sn=sn, padding='same', **kwargs)(inputs) #[b, h, w, C]
    #key = MaxPool2D(pool_size=2, strides=2, padding='same')(key)
    
    value = Convolution(channels, 1, sn=sn, padding='same', **kwargs)(inputs)  #[b, h, w, c]
    #value = MaxPool2D(pool_size=2, strides=2, padding='same')(value)

    query = tf.reshape(query, [b, -1, channels]) #[b, N/4, C]
    key = tf.reshape(key, [b, -1, channels])    #[b, N/4, C]
    value = tf.reshape(value, [b, -1, channels])        #[b, N/4, c//2]
    attention = tf.nn.softmax(tf.linalg.matmul(key, query, transpose_b=True)) #[b, N/4, N/4] #N x N/4
    out = tf.reshape(tf.linalg.matmul(attention, value), [b, h, w, channels]) #[b, N/4, c//2] -> [b, h//2, w//2, c//2]  
    #out = Conv2DTranspose(filters=c, kernel_size=1, strides=2, padding='same')(out)
    out = Convolution(c, 1, sn=sn, padding='same', **kwargs)(out)
    #out = UpSampling2D(2, interpolation='nearest')(out)
    out = Scale()(out) + inputs
    return out


def BConv2D(inputs, filters, kernel=3, strides=1,
            padding='same', kernel_initializer='glorot_uniform', alpha=0.2):
    x = Conv2D(filters=filters, kernel_size=kernel, strides=strides, padding=padding,
               kernel_initializer=kernel_initializer)(inputs)
    x = BatchNormalization()(x)
    return relu(x, alpha=alpha)


def LConv2D(inputs, filters, kernel=3, strides=1,
            padding='same', kernel_initializer='glorot_uniform', alpha=0.2):
    x = Conv2D(filters=filters, kernel_size=kernel, strides=strides, padding=padding,
               kernel_initializer=kernel_initializer)(inputs)
    x = LayerNormalization()(x)
    return relu(x, alpha=alpha)


