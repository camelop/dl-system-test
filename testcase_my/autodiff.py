#import tensorflow as tf
import tensorwolf as tf
import numpy as np

# create model
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
muln = tf.matmul(x, W) + b
#y = tf.nn.softmax(muln)

expn = tf.exp(muln)
#redn = tf.broadcastto_op(tf.reduce_sum(expn2, axis=-1, keep_dims=True), expn)
redn = tf.broadcastto_op(tf.reduce_sum(expn, axis=-1), expn)
y = expn / redn


# define loss and optimizer
y_ = tf.placeholder(tf.float32, [None, 10])

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ *
                                              tf.log(y), reduction_indices=[1]))

#W_grad = tf.gradients(cross_entropy, [W])[0]
W_grad = tf.gradients(cross_entropy, [W])[0]
train_step = tf.assign(W, W - 0.5 * W_grad)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# get the mnist dataset (use tensorflow here)
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# train
for _ in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    ww, g, ce = sess.run([train_step, W_grad, cross_entropy],
                         feed_dict={x: batch_xs, y_: batch_ys})
    print(np.sum(ce))
# eval
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

ans = sess.run(accuracy, feed_dict={
               x: mnist.test.images, y_: mnist.test.labels})

print("Accuracy: %.3f" % ans)
assert ans >= 0.87
