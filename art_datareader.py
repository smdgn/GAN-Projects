import tensorflow as tf
import os

def _get_file_paths(root):
    base = os.path.join(root, "*.tfrecord.*")
    return tf.data.Dataset.list_files(base, shuffle=True)

def _convert_frame_data(data):
    decoded_frame = tf.image.decode_image(data)
    return tf.cast(decoded_frame, tf.float32) / 255



class DataReader(object):
    def __init__(self, batch_size, root):
        #self._frame_height = 128
        #self._frame_width = 384
        #self._channels = 3
        with tf.device('/cpu'):
            filenames = _get_file_paths(root)
            dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=os.cpu_count())
            dataset = dataset.map(self._parse_function, num_parallel_calls=os.cpu_count())
            dataset = dataset.batch(batch_size, drop_remainder=True)
            self.data = dataset

    
    def _parse_function(self, data):
        datadict = {
            'frame': tf.io.FixedLenFeature([], dtype = tf.string),
            'size' : tf.io.FixedLenFeature([3], dtype = tf.int64)
        }
        example = tf.io.parse_single_example(data, datadict)
        frames = self._process_frames(example)
        return frames
    

    def _process_frames(self, example):
        frame = _convert_frame_data(example['frame'])
        size = example['size']
        frame = tf.reshape(frame, [size[0], size[1], size[2]])
        frame = tf.image.resize(frame, (448, 448))
        return frame

