from keras.utils import np_utils
from keras.datasets import cifar10
import tensorflow as tf
import time
from matplotlib import pyplot as plt

from PIL import Image

(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()



plt.imshow(train_images[0])
plt.show()

#https://gruuuuu.github.io/machine-learning/cifar10-cnn/#