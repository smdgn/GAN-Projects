import cv2
import tfutils
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


root = r"xxx\train"
write_dir = r"xxx\art_train.tfrecord"

image_path = tfutils.retdir(root)
partwriter = tfutils.tf_part_writer(write_dir, 1)

for path in image_path:
    try:
        size, frame = tfutils.load_image(path, '.jpg')

        datadict = {
            'frame': tfutils.bytes_feature([frame]),
            'size' : tfutils.int_feature(size)
            }

        example = tf.train.Example(features = tf.train.Features(feature = datadict))
        partwriter.write_part(example)
    except:
        print(path)
