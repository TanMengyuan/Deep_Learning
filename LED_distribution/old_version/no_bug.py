import tensorflow as tf
import numpy as np


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

led_matrix = tf.zeros(shape=[20, 20], dtype=tf.float32)
rec_matrix = tf.ones(shape=[20, 20], dtype=tf.float32)

xs = tf.placeholder(dtype=tf.float32, shape=[20, 20], name='input_x')
ys = tf.placeholder(dtype=tf.float32, shape=[20, 20], name='output_y')

W_conv1 = weight_variable([20 * 20, 1024])
b_conv1 = bias_variable([1024])

x_input = tf.reshape(xs, [1, 20 * 20])
h_conv1 = tf.nn.relu(tf.matmul(x_input, W_conv1) + b_conv1)

W_conv2 = weight_variable([1024, 20 * 20])
b_conv2 = bias_variable([20 * 20])

pred = tf.nn.relu(tf.matmul(h_conv1, W_conv2) + b_conv2)

y_output = tf.reshape(pred, [20, 20])
cross_entropy = -tf.reduce_sum(rec_matrix * tf.log(pred))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
avg = tf.reduce_mean(pred)

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())