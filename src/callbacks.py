import os
from pathlib import Path
from typing import Union

import numpy as np
import tensorflow as tf
from tensorflow import keras

from utils import cast_img


class TensorboardGAN(keras.callbacks.Callback):
    """
    Write generated sample images to tensorboard. Calls model.generate(samples=max_outputs)

    Args:
        log_dir: (str) main directory to store the logs
        log_freq: (int) specifying to log every n train steps
        max_outputs: (int) maximum generated batch size
        folder_name: (str) folder named appended to the logging directory

    Raises:
        ValueError: if no generate function is defined for the keras model

    """
    def __init__(self,
                 log_dir: str,
                 log_freq: int = 1,
                 max_outputs: int = 10,
                 folder_name: str = 'Generated_Images'):
        super(TensorboardGAN, self).__init__()
        # Setup
        if getattr(self.model, 'generate', None) is None:
            raise ValueError("No generate function defined in model")

        self.writing_dir = log_dir
        self.log_freq = log_freq
        self.max_outputs = max_outputs
        self.folder_name = folder_name
        # Path
        self._absolute_path = Path(self.writing_dir).joinpath(self.folder_name)
        self.writer = tf.summary.create_file_writer(str(self._absolute_path))
        # States
        self._epoch = 0
        self._step = 0

    def _plot_images(self):
        if self._step % self.log_freq == 0:
            # check if return value has also labels or images only
            generated_images = self.model.generate(training=False, samples=self.max_outputs)
            # TODO: checking GAN Type is not elegant this way
            generated_images = generated_images[0] if isinstance(generated_images, tuple) else generated_images
            # images are in range [-1,1]
            generated_images = cast_img(generated_images)
            with self.writer.as_default():
                tf.summary.image("Generated Samples", generated_images,
                                 step=self._test_step, max_outputs=self.max_outputs)
        self._test_step += 1

    def on_train_begin(self, logs=None):
        self._plot_images()


class TensorboardProjector(keras.callbacks.Callback):
    def __init__(self,
                 log_dir: str,
                 data: Union[tf.data.Dataset, str],
                 labels: Optional[Sequence] = None,
                 encoder_name: str = 'encoder',
                 batch_samples: int = 100,
                 total_samples: int = 1000):
        """
        Writes model activations of the given layer as embeddings into a checkpoint and .tsv file. The feature maps are
        flattened and written per-point, with the last dimension encoding the feature dimension e.g the number of features
        the encoding contains per point. The labels are converted to strings if a conversion table is given, otherwise
        the provided groundtruth data is used as metafile. Beware that a high number of batch samples will increase
        memory consumption significantly, but executes faster.
         Checkpoints are written to log_dir/validation and .tsv files to log_dir/embeddings

        Args:
            log_dir: target directory where logs are written to
            data: a tf.data.Dataset instance containing images or str to the image directory
            labels: conversion table, should be in the cityscapes label format
            encoder_name: name of the layer to take the embeddings from
            batch_samples: number of samples generated per batch
            total_samples: total number of samples to generate
        """
        super(TensorboardProjector, self).__init__()
        # setup directories and paths
        self.log_dir = Path(log_dir)
        self.embeddings_dir = self.log_dir.joinpath('embeddings')
        self.validation_dir = self.log_dir.joinpath('validation')
        self.feature_path = str(self.embeddings_dir.joinpath('feature_vector.tsv'))
        self.meta_path = str(self.embeddings_dir.joinpath('metadata.tsv'))
        self.sprite_path = str(self.embeddings_dir.joinpath('sprite.jpg'))
        os.makedirs(self.embeddings_dir, exist_ok=True)
        os.makedirs(self.validation_dir, exist_ok=True)

        self.data = data
        self.encoder_name = encoder_name
        self.batch_samples = batch_samples
        self.total_samples = total_samples
        if labels is not None:
            # create Hashtable
            keys, values = zip(*[(label.trainId, label.name) for label in labels if label.trainId < 255])
            self.table = tf.lookup.StaticHashTable(
                tf.lookup.KeyValueTensorInitializer(tf.constant(keys), tf.constant(values)),
                default_value='ignored')
            self.fmt = '%s'
        else:
            self.table = None
            self.fmt = '%d'

        # internals
        self._rounds = int(self.total_samples / self.batch_samples)
        self._sprite_size = (100, 100)  # (height, width)
        self._square_count = int(np.ceil(np.sqrt(self._rounds*self.batch_samples)))
        self._master_size = tuple(size*self._square_count for size in self._sprite_size)
        self._sprite = np.zeros((*self._master_size, 3), np.uint8)

    def on_train_end(self, logs=None):
        # Create the directories


        features = []

        # generate the images
        with open(self.feature_path, 'a') as feature_file, open(self.meta_path, 'at') as meta_file:
            for round in range(self._rounds):
                generated_images = self.model.generate(training=False, samples=self.batch_samples)
                # TODO: checking GAN Type is not elegant this way
                generated_images = generated_images[0] if isinstance(generated_images, tuple) else generated_images
                # get the last latent vector used
                embedding = self.model.z
                #channels = embedding.shape[-1]
                # resize to sprite size eg 100x100
                generated_images = tf.image.resize(generated_images, self._sprite_size,
                                                   method=tf.image.ResizeMethod.BILINEAR)
                #embedding = tf.reshape(embedding, (self.batch_samples, -1))

                np.savetxt(feature_file, embedding.numpy(), delimiter='\t', fmt='%1.16f')

                #if self.table is not None:
                    #y = self.table[y]
                #for single_emb in embedding:
                    # Write metadata and vectors to .tsv file
                    #np.savetxt(feature_file, single_emb.numpy(), delimiter='\t', fmt='%1.16f')
                    #np.savetxt(meta_file, single_label.numpy(), delimiter='\n', fmt=self.fmt)

                features.append(tf.cast(tf.reshape(embedding, shape=(-1, c)), tf.float16))
        # Write vectors to checkpoint
        embd = tf.concat(features, axis=0)
        embd = tf.Variable(embd)
        checkpoint = tf.train.Checkpoint(embedding=embd)
        checkpoint.save(os.path.join(self.validation_dir, "embedding.ckpt"))

        # Setup projector config
        config = projector.ProjectorConfig()
        embedding = config.embeddings.add()
        embedding.tensor_name = "embedding/.ATTRIBUTES/VARIABLE_VALUE"
        embedding.metadata_path = self.meta_path
        projector.visualize_embeddings(self.validation_dir, config)
