import numpy as np
from utils import xe_init


class Dense:
    def __init__(self, in_dim, out_dim, activation=np.tanh, lr=1e-3):
        self.W = xe_init((out_dim, in_dim))
        self.b = np.zeros((1, out_dim))
        self.activation = activation
        self.lr = lr
        if activation == np.tanh:
            self.d_activation = lambda x: 1 - np.tanh(x)**2
        elif activation is not None:
            self.d_activation = lambda x: (x>0).astype(float)
        else:
            self.d_activation = None

    def forward(self, x):
        self.x = x
        self.z = x @ self.W.T + self.b
        if self.activation is not None:
            return self.activation(self.z)
        return self.z

    def backward(self, delta):
        if self.d_activation is not None:
            delta *= self.d_activation(self.z)
        dW = delta.T @ self.x / self.x.shape[0]
        db = delta.sum(axis=0, keepdims=True) / self.x.shape[0]
        dx = delta @ self.W
        self.W -= self.lr * dW
        self.b -= self.lr * db
        return dx
