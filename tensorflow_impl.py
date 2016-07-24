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

# Implementing the convolution neural network
# Reshape in accordance to convnet input (4D tensor)
x_image = tf.reshape(x, [-1, 28, 28, 1])

# First conv and pool layer
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# Second conv and pool layer
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# Densely connected layer (Fully Connected - FC)
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1), b_fc1)

# Dropout to reduce overfitting
keep_prob = tf.placeholder(dtype=tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# Readout layer
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
