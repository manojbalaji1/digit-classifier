import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('data/mnist_ubyte_gz',
                                  one_hot=True,
                                  dtype=tf.float32)

session = tf.InteractiveSession()
x = tf.placeholder(dtype=tf.float32, shape=[None, 784])
y_ = tf.placeholder(dtype=tf.float32, shape=[None, 10])
