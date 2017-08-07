import numpy as np


def a(input):
    sum = 0
    for i in range(5000):
        sum += i
    sum = sum * 100
    input[0][0] = 1


def b():
    s = np.zeros((2, 2))
    a(s)
    return s[0][0]


print(b())


'''
a = np.ones((2, 2, 2))
print(a)
print(np.argmax(a, axis=2))
a.flat[np.argmax(a, axis=2)] = 250
print(a)
'''
