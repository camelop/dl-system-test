import numpy as np
import tensorwolf as tf
i = np.ones((1, 3, 3, 2))
g = np.ones((1, 3, 3, 2))
o = np.ones((2, 2, 2, 2))
ret = tf.c_ops.conv2d_filter_gradient(input=i, gradient=g, ori_filter=o)
print("i:", i)
print("g:", g)
print("o:", o)
print("ret:", ret)
