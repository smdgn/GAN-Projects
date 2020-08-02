import tensorflow as tf
import numpy as np
import os
import matplotlib.image as mpimg
import io
import cv2
import itertools
import json

#-------tf-tools------------#
def bytes_feature(x):
       return tf.train.Feature(bytes_list=tf.train.BytesList(value=x))

def float_feature(x):
       return tf.train.Feature(float_list=tf.train.FloatList(value=x))

def int_feature(x):
       return tf.train.Feature(int64_list=tf.train.Int64List(value=x))

class tf_part_writer():
    
    def __init__(self, path, splits):
        self.__writecount = 0
        self.__partnum = 0
        self.__path = path
        self.__splits = splits
        self.__writer = self.__initWriter(path)
    
    def get_writecount(self):
        return self.__writecount
    
    def get_partnumber(self):
        return self.__partnum

    def __initWriter(self, path):
        partpath = path +'.{:03d}'.format(self.__partnum)
        return tf.python_io.TFRecordWriter(partpath)   

    def write_part(self, data, printable = True):
        """Writes a tfrecord file in number of  '.__splits' """ 
        if self.__writecount == 0 and self.__partnum != 0:
            self.__writer.close()
            partpath = self.__path+'.{:03d}'.format(self.__partnum)
            self.__writer = tf.python_io.TFRecordWriter(partpath)   
        self.__writer.write(data.SerializeToString())
        self.__writecount += 1
        if printable==True:
            print('writing sample {} of filenr: {} '.format(self.__writecount, self.__partnum))
        if self.__writecount == self.__splits:
            self.__partnum += 1
            self.__writecount = 0
    
    def __del__(self):
        self.__writer.close()

#--------image-tools------#

def readImage(path):  #decode with decodeImage
    with open(path, "rb") as f:
        return f.read()

def decodeImage(img, format):
    return mpimg.imread(io.BytesIO(np.fromstring(img, dtype='uint8')), format)


def load_image(path, encoding, size=None):  #image encoding via cv2, decode via readcv2images
    img = cv2.imread(path)
    if size is not None:
        img = cv2.resize(img, size)
    return img.shape, cv2.imencode(encoding, img)[1].tostring()

def readcv2images(image):
    image = np.fromstring(image, dtype= 'uint8')
    image = cv2.imdecode(image, cv2.IMREAD_UNCHANGED)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


#--------directory/path-manipulation----------#

def retdir(dir, absolute_path=True):
    if absolute_path == True:
        return [os.path.join(dir, file) for file in sorted(os.listdir(dir))]
    else:
        return sorted(os.listdir(dir))

def walkdir(root):
    path = []
    files = []
    for dirlist, subdirlist, filelist in os.walk(root):
        if subdirlist.__len__() == 0:
            path.append(dirlist)
        if filelist.__len__() != 0:
            files.append(sorted(filelist))
    return path, files

def getTotalPaths(pathlist, filelist):
    """join files and dirs from walkdir. One pathlist-entry contains N files."""
    """ Returns list of every imagepath in the walked dir"""
    return [os.path.join(path, files) for path, subdir in zip(pathlist, filelist) for files in subdir]


#-----------misc-------------#

def grouper(iterable, n, fillvalue=None):
    """Collect data into fixed-length chunks or blocks"""
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return itertools.zip_longest(*args, fillvalue=fillvalue)

def readJson(file):
    if file.endswith('.json'):
        with open(file, 'r') as f:
            return json.load(f)

def writeFrames(path, rootdir):
    """reads Frames from Input Video  and saves them as jpg in rootdir"""
    head, filename  = os.path.split(path)
    filename = os.path.splitext(filename)[0]
    folder = os.path.basename(head)
    folderpath = os.path.join(rootdir, folder)
    folderfile = os.path.join(folderpath, filename)

    if not os.path.exists(folderpath):
        os.makedirs(folderpath)
        print("Directory " , folderpath ,  " Created ")
    else:    
        print("Directory " , folderpath ,  " already exists")
    os.mkdir(folderfile)

    vidcap = cv2.VideoCapture(path)
    success,image = vidcap.read()
    count = 0
    while success:
        cv2.imwrite(os.path.join(folderfile, "frame{:03d}.jpg".format(count)), image)
        success, image = vidcap.read()
        print("Read a new frame: ", success)
        count += 1
