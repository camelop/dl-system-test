import numpy as np
a = np.random.normal(size=(2, 2, 2))
a = np.ones((2, 2, 2))
print(a)
print(np.argmax(a, axis=2))
a.flat[np.argmax(a, axis=2)] = 250
print(a)
