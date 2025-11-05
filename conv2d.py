import numpy as np
from numpy.lib._stride_tricks_impl import as_strided

from utils import xe_init


def im2col(x, k_h, k_w, stride):
    """
    Преобразует 4D вход (B, C, H, W) в 3D "матрицу патчей" (B, N, C*k_h*k_w)
    где N = H_out * W_out
    """
    B, C, H, W = x.shape
    H_out = (H - k_h) // stride + 1
    W_out = (W - k_w) // stride + 1

    # Используем strided view (без циклов)
    shape = (B, C, H_out, W_out, k_h, k_w)
    strides = (
        x.strides[0],
        x.strides[1],
        x.strides[2] * stride,
        x.strides[3] * stride,
        x.strides[2],
        x.strides[3],
    )
    patches = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)
    cols = patches.transpose(0, 2, 3, 1, 4, 5).reshape(B, H_out * W_out, -1)
    return cols, H_out, W_out


def col2im(cols, x_shape, k_h, k_w, stride):
    """
    Обратная операция im2col — собирает градиент dX из dcols.
    """
    B, C, H, W = x_shape
    H_out = (H - k_h) // stride + 1
    W_out = (W - k_w) // stride + 1

    dx = np.zeros(x_shape)
    dcols_reshaped = cols.reshape(B, H_out, W_out, C, k_h, k_w).transpose(0, 3, 1, 2, 4, 5)

    for i in range(H_out):
        for j in range(W_out):
            h_start = i * stride
            w_start = j * stride
            dx[:, :, h_start:h_start + k_h, w_start:w_start + k_w] += dcols_reshaped[:, :, i, j, :, :]
    return dx


class Conv2D:
    def __init__(self, out_channels, in_channels, kernel_size, stride=1, padding=0, activation=np.tanh, lr=1e-3):
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.activation = activation
        self.d_activation = None
        if activation == np.tanh:
            self.d_activation = lambda x: 1 - np.tanh(x) ** 2
        elif activation is not None:
            self.d_activation = lambda x: (x > 0).astype(float)  # ReLU derivative
        self.lr = lr
        self.W = xe_init((out_channels, in_channels, kernel_size, kernel_size))
        self.b = np.zeros((out_channels, 1))

    def _im2col(self, x):
        B, C, H, W = x.shape
        k = self.kernel_size
        s = self.stride
        H_out = (H - k) // s + 1
        W_out = (W - k) // s + 1

        shape = (B, C, H_out, W_out, k, k)
        strides = (
            x.strides[0],
            x.strides[1],
            x.strides[2] * s,
            x.strides[3] * s,
            x.strides[2],
            x.strides[3]
        )
        windows = as_strided(x, shape=shape, strides=strides)
        cols = windows.transpose(0, 2, 3, 1, 4, 5).reshape(B, H_out * W_out, C * k * k)
        return cols, H_out, W_out

    def forward(self, x):
        if self.padding > 0:
            x = np.pad(x, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')
        self.x = x
        self.cols, self.H_out, self.W_out = self._im2col(x)
        W_col = self.W.reshape(self.out_channels, -1)
        z = self.cols @ W_col.T + self.b.T  # (B, H_out*W_out, out_channels)
        z = z.transpose(0, 2, 1).reshape(x.shape[0], self.out_channels, self.H_out, self.W_out)
        self.z = z
        return self.activation(z)

    def backward(self, delta):
        delta *= self.d_activation(self.z)
        B, F, H_out, W_out = delta.shape
        delta_flat = delta.reshape(B, F, -1)  # (B, F, N)
        W_col = self.W.reshape(F, -1)

        # dW
        dW = np.matmul(delta_flat, self.cols)  # (B, F, C*k*k)
        dW = dW.sum(axis=0).reshape(self.W.shape) / B
        # db
        db = delta_flat.sum(axis=(0, 2)).reshape(self.b.shape) / B

        # dx
        dcols = np.matmul(delta_flat.transpose(0, 2, 1), W_col)  # (B, N, C*k*k)
        dcols = dcols.reshape(B, H_out, W_out, self.in_channels, self.kernel_size, self.kernel_size)
        dcols = dcols.transpose(0, 3, 1, 2, 4, 5)

        dx = np.zeros_like(self.x)
        for i in range(H_out):
            for j in range(W_out):
                h_start, w_start = i * self.stride, j * self.stride
                dx[:, :, h_start:h_start + self.kernel_size, w_start:w_start + self.kernel_size] += dcols[:, :, i, j, :,
                                                                                                    :]

        # обновляем параметры
        self.W -= self.lr * dW
        self.b -= self.lr * db
        return dx
