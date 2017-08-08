import numpy as np

b = np.array([0.8])
a = np.random.uniform(size=(5, 5))
c = (a < b).astype(np.float32)
print(a)
print(c)
