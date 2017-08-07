import numpy as np
from scipy import signal
import c_ops
a = np.random.randint(low=0, high=2, size=(1, 8, 8, 1))
b = np.random.randint(low=0, high=2, size=(4, 4, 1, 1))
c = signal.correlate2d(
    a.reshape((8, 8)), b.reshape((4, 4)), 'valid').astype(int)
cc = c_ops.correlate2d(a, b, [1, 1, 1, 1], 'VALID').reshape(5, 5).astype(int)
print(a.reshape(8, 8))
print()
print(b.reshape(4, 4))
print("should be")
print(c)
print("and mine:")
print(cc)
'''
a = np.ones((2, 2, 2))
print(a)
print(np.argmax(a, axis=2))
a.flat[np.argmax(a, axis=2)] = 250
print(a)
'''
