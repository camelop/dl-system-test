import numpy as np


a = np.ones([2, 3, 4])
b = np.sum(a, axis=1, keepdims=True)
print(b.shape)
