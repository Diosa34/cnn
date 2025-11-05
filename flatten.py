import numpy as np


class Flatten:
    def __init__(self):
        self.orig_shape = None

    def forward(self, x):
        self.orig_shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, delta):
        return delta.reshape(self.orig_shape)