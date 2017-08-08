import tensorwolf as tf
import numpy as np

shape = (100, 100)
x = tf.placeholder(shape=shape, dtype=tf.float32)
prob = tf.placeholder(dtype=tf.float32)
test = tf.nn.dropout(x, prob)

print(np.mean(test.run(feed_dict={x: np.ones(shape), prob: 0.8
                                  })))
