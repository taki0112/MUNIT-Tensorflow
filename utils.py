import tensorflow as tf
from tensorflow.contrib import slim
from scipy import misc
import os
import numpy as np

# https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/
# https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/
class ImageData:

    def __init__(self, load_size, channels):
        self.load_size = load_size
        self.channels = channels

    def image_processing(self, filename):
        x = tf.read_file(filename)
        x_decode = tf.image.decode_jpeg(x, channels=self.channels)
        img = tf.image.resize_images(x_decode, [self.load_size, self.load_size])
        img = tf.cast(img, tf.float32) / 127.5 - 1

        return img

def prepare_data(dataset_name, size):
    data_path = os.path.join("./dataset", dataset_name)

    trainA = []
    trainB = []
    for path, dir, files in os.walk(data_path):
        for file in files:
            image = os.path.join(path, file)
            if path.__contains__('trainA') :
                trainA.append(misc.imresize(misc.imread(image, mode='RGB'), [size, size]))
            if path.__contains__('trainB') :
                trainB.append(misc.imresize(misc.imread(image, mode='RGB'), [size, size]))


    trainA = preprocessing(np.asarray(trainA))
    trainB = preprocessing(np.asarray(trainB))

    np.random.shuffle(trainA)
    np.random.shuffle(trainB)

    return trainA, trainB

def load_test_data(image_path, size=256):
    img = misc.imread(image_path, mode='RGB')
    img = misc.imresize(img, [size, size])
    img = np.expand_dims(img, axis=0)
    img = preprocessing(img)

    return img

def preprocessing(x):
    x = x/127.5 - 1 # -1 ~ 1
    return x

def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)

def inverse_transform(images):
    return (images+1.) / 2

def imsave(images, size, path):
    return misc.imsave(path, merge(images, size))

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[h*j:h*(j+1), w*i:w*(i+1), :] = image

    return img

def show_all_variables():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)

def check_folder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir
