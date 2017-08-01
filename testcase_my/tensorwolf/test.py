import numpy as np
a = np.ones((3, 4, 5))
b = np.zeros((3, 4, 5))
b[0:2, 0:3, 1:4] = a[0:2, 0:3, 1:4]
print(b)
