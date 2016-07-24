import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('data/mnist_ubyte_gz',
                                  one_hot=True,
                                  dtype=tf.float32)

session = tf.InteractiveSession()
x = tf.placeholder(dtype=tf.float32, shape=[None, 784])
y_ = tf.placeholder(dtype=tf.float32, shape=[None, 10])


def weight_variable(shape):
    initial = tf.truncated_normal(shape=shape, mean=0.0, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(value=.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(
        input=x, filter=W, strides=[1, 1, 1, 1], padding='SAME'
    )


def max_pool_2x2(x):
    return tf.nn.max_pool(
        value=x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME'
    )
