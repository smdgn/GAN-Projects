import tensorflow as tf
import os
import cv2

def cast_img(x):
    x = (x + 1.) * 127.5
    return tf.cast(tf.clip_by_value(x, 0, 255), tf.uint8)


def sorted_dir(dir_path, absolute_path=True):
    if absolute_path:
        return [os.path.join(dir_path, file) for file in sorted(os.listdir(dir_path))]
    else:
        return sorted(os.listdir(dir_path))


def load_image(path, encoding, size=None):
    img = cv2.imread(path)
    if size is not None:
        img = cv2.resize(img, size)
    return img.shape, cv2.imencode(encoding, img)[1].tostring()


def bytes_feature(x):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[x]))


def float_feature(x):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[x]))


def int_feature(x):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[x]))


class TfPartWriter:

    def __init__(self, path, splits):
        self.writecount = 0
        self.partnum = 0
        self.path = path
        self.splits = splits
        self.writer = self._init_writer(path)

    def _init_writer(self, path):
        partpath = path + '.{:03d}'.format(self.partnum)
        return tf.io.TFRecordWriter(partpath)

    def write_part(self, data, verbose=True):
        """Writes a tfrecord file in number of  '.__splits' """
        if self.writecount == 0 and self.partnum != 0:
            self.writer.close()
            partpath = self.path + '.{:03d}'.format(self.partnum)
            self.writer = tf.io.TFRecordWriter(partpath)
        self.writer.write(data.SerializeToString())
        self.writecount += 1
        if verbose:
            print('writing sample {} of filenr.: {} '.format(self.writecount, self.partnum))
        if self.writecount == self.splits:
            self.partnum += 1
            self.writecount = 0

    def __del__(self):
        self.writer.close()


class DataReader(object):
    def __init__(self, batch_size, root, output_shape=None):
        self.output_shape = output_shape
        self.root = root
        with tf.device('/cpu'):
            filenames = self._get_file_paths()
            self.data = tf.data.TFRecordDataset(filenames, num_parallel_reads=os.cpu_count()) \
                .shuffle(10000, reshuffle_each_iteration=True) \
                .batch(batch_size, drop_remainder=True) \
                .map(self._parse_function, num_parallel_calls=os.cpu_count()) \
                .cache() \
                .prefetch(tf.data.experimental.AUTOTUNE)

    def _get_file_paths(self):
        base = os.path.join(self.root, "*.tfrecord.*")
        return tf.data.Dataset.list_files(base, shuffle=True)

    def _parse_function(self, data):
        datadict = {
            'frame': tf.io.FixedLenFeature([], dtype=tf.string),
            'class': tf.io.FixedLenFeature([], dtype=tf.string),
            'nr': tf.io.FixedLenFeature([], dtype=tf.int64),
        }
        example = tf.io.parse_example(data, datadict)
        frames = self._process_frames(example)
        #return frames, example['class'], example['nr']
        return frames, example['nr']

    def _process_frames(self, example):
        def decode(frame):
            return tf.io.decode_raw(frame, 'uint8')
            
        frame = tf.map_fn(decode, example['frame'], dtype=tf.uint8)
        frame = (tf.cast(frame, tf.float32) / 128.) - 1.
        #size = example['size']
        #frame = tf.reshape(frame, [size[0], size[1], size[2]])
        frame = tf.reshape(frame, [-1, 64,64,3])
        if self.output_shape is not None:
            frame = tf.image.resize(frame, self.output_shape)
        return frame
    

