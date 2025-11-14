import numpy as np


class AvgPool2D:
    def __init__(self, kernel_size=2, stride=2):
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):
        self.x_shape = x.shape
        B, C, H, W = x.shape
        k, s = self.kernel_size, self.stride
        H_out = (H - k)//s + 1
        W_out = (W - k)//s + 1
        out = np.zeros((B, C, H_out, W_out))
        for i in range(H_out):
            for j in range(W_out):
                out[:, :, i, j] = np.mean(x[:, :, i*s:i*s+k, j*s:j*s+k], axis=(2, 3))
        return out

    def backward(self, delta):
        B, C, H_out, W_out = delta.shape
        k, s = self.kernel_size, self.stride
        dx = np.zeros(self.x_shape)
        for i in range(H_out):
            for j in range(W_out):
                dx[:, :, i*s:i*s+k, j*s:j*s+k] += delta[:, :, i, j][:, :, None, None]  # /(k*k)
        return dx
