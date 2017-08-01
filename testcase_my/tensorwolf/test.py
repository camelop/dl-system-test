import tensorflow as tf
import numpy as np
a = np.ones((1, 5, 5, 1))

inputs = tf.placeholder(tf.float32, [1, 5, 5, 1])
f = tf.Variable(tf.ones((3, 3, 1, 1)))
conv = tf.nn.conv2d(inputs, f, strides=[
                    1, 1, 1, 1], padding='SAME')

pool = tf.nn.max_pool(conv, [1, 2, 2, 1], [1, 2, 2, 1], "SAME")
#re = tf.reshape(conv, [5, 5])
re = tf.reshape(pool, [3, 3])
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    print(re.eval(feed_dict={inputs: a}))
