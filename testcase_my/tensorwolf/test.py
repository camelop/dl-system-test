import numpy as np


class myint(object):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return '$' + str(self.value) + '$'

    def __sub__(self, rhs):
        print(self, '-', rhs)

    def __rsub__(self, lhs):
        print(lhs, '-', self)


a = myint(5)
b = myint(3)
c = a - b
