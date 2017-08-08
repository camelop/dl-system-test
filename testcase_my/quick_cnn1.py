""" import your model here """
import tensorflow as tf
#import tensorwolf as tf
import numpy as np
""" your model should support the following code """


def weight_variable(shape):
    initial = tf.random_normal(shape, stddev=0.1)
    initial = np.ones(shape).astype(np.float32) * 0.01
    initial[:, 0, ...] = -initial[:, 0, ...]
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                          padding='SAME')


# input
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

magic_number = 1
# first layer
W_conv1 = weight_variable([5, 5, 1, magic_number])
b_conv1 = bias_variable([magic_number])
x_image = tf.reshape(x, [-1, 28, 28, 1])

temp1 = conv2d(x_image, W_conv1)
temp0 = temp1 + b_conv1

h_conv1 = tf.nn.relu(temp0)
h_pool1 = max_pool_2x2(h_conv1)

# second layer
W_conv2 = weight_variable([5, 5, magic_number, magic_number * 2])
b_conv2 = bias_variable([magic_number * 2])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# densely connected layer
W_fc1 = weight_variable([7 * 7 * magic_number * 2, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * magic_number * 2])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# readout layer
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

temp4 = tf.matmul(h_fc1, W_fc2)  # something wrong here
y_conv = temp4 + b_fc2

# loss
temp3 = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv)
cross_entropy = tf.reduce_mean(temp3)

train_step = tf.train.GradientDescentOptimizer(1e-2).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

grad = tf.gradients(cross_entropy, [W_conv2])[0]

# Get the mnist dataset (use tensorflow here)
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# train and eval
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(1):
        batch = mnist.train.next_batch(2, shuffle=False)
        if i % 50 == 33:
            train_accuracy = accuracy.eval(feed_dict={x: batch[0],
                                                      y_: batch[1]})
            print('Step %d, trainning accuracy %g' % (i, train_accuracy))
        p = sess.run([grad], feed_dict={
                     x: batch[0], y_: batch[1]})[0]
        np.set_printoptions(threshold=np.nan)
        # print(p)
        print(np.sum(np.abs(p)))
'''
    ans = accuracy.eval(feed_dict={x: mnist.test.images,
                                   y_: mnist.test.labels})
    print('Test accuracy: %g' % ans)
    assert ans > 0.88
'''
