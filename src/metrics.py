import numpy as np
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from numpy import asarray
from numpy.random import shuffle
from scipy.linalg import sqrtm
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.datasets.mnist import load_data
from skimage.transform import resize
from keras.datasets import cifar10
import matplotlib.pyplot as plt
import tensorflow as tf

class IntraFID:
    def __init__(self, input_shape=(75,75,3), num_samples=20):
        
        self.input_shape = input_shape
        self.num_samples = num_samples
        self.model = InceptionV3(include_top=False, pooling='avg', input_shape=input_shape)
        #allocate empty array
        #10 classes, 20 samples per class, wxhxc
        self.image_chunks = np.zeros((10, self.num_samples, *self.input_shape), dtype=np.float32)
        self.initial= True
        
    def _scale_images(self, images):
        return asarray([resize(image.astype('float32'), self.input_shape,0) for image in images])
        #return tf.image.resize(tf.cast(images, tf.float32), (self.input_shape[0], self.input_shape[1]), method='nearest')
    
    def _scale_images_2(self, images):
        #return tf.image.resize(tf.cast(images, tf.float32), (self.input_shape[0], self.input_shape[1]), method='nearest')
        return asarray([resize(tf.cast(image, tf.float32), self.input_shape,0) for image in images])
    
    def _calculate_fid(self, real_images, fake_images):
        #calculate activations
        act1 = self.model.predict(real_images)
        act2 = self.model.predict(fake_images)
        
        #calculate mean and covariance statistics
        mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
        mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)
        
        # calculate sum squared difference between means
        ssdiff = np.sum((mu1 - mu2)**2.0)
        
        # calculate sqrt of product between cov
        covmean = sqrtm(sigma1.dot(sigma2))
        
        # check and correct imaginary numbers from sqrt
        if iscomplexobj(covmean):
            covmean = covmean.real
        #calculate score
        fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
        return fid
    
    def _load_data(self):
        #load data
        (real_images, real_labels), (_,_) = cifar10.load_data()
        
        image_chunk = np.zeros((10, self.num_samples, 32,32,3), dtype=np.uint8)
        #sort labels
        real_labels_index = np.argsort(real_labels.flatten())
        for i in range(10):
               image_chunk[i,:,:,:,:] = real_images[real_labels_index[5000*i:5000*i+self.num_samples], :,:,:]
        self.image_chunks = [preprocess_input(self._scale_images(images)) for images in image_chunk]
        self.initial = False
        
        
    def get_fid(self, fake_images):
        
        fake_images = [self._scale_images_2(images) for images in fake_images]
        
        if self.initial:
            self._load_data()
        
        class_fid = [self._calculate_fid(real, fake) for real,fake in zip(self.image_chunks, fake_images)]
        return class_fid, np.mean(class_fid)
    
    
def sample_ssim(model, n_classes, z_dim, n_samples=100, y=None):
    if y is None:
        y = [tf.fill([n_samples, 1], i) for i in range(n_classes)]
        ssim_list = []
        for labels in y:
            z = tf.random.normal((n_samples, z_dim))
            images = model([z,y], training=False)
            images = (images + 1.0)* 0.5
            random_indexes = tf.random.uniform(shape=(n_samples, 2), minval=0, maxval=n_samples, dtype=tf.int64)
            ssim = [tf.image.ssim_multiscale(images[index1,...], images[index2,...], 1.0, power_factors= (0.0448, 0.2856, 0.3001), filter_size=7) for index1,index2 in random_indexes]
            ssim_list.append(tf.reduce_mean(ssim))
        return ssim_list
    else:
        assert y < self.n_classes
        y = tf.fill([n_samples, 1], y)
        z = tf.random.normal((n_samples, z_dim))
        images = model([z,y], training=False)
        images = (images + 1.0)* 0.5
        random_indexes = tf.random.uniform(shape=(n_samples, 2), minval=0, maxval=n_samples, dtype=tf.int64)
        ssim = [tf.image.ssim_multiscale(images[index1,...], images[index2,...], 1.0, power_factors= (0.0448, 0.2856, 0.3001), filter_size=7) for index1,index2 in random_indexes]
        return tf.reduce_mean(ssim)
        
    
    


    

        
