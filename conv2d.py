import numpy as np

from utils import xe_init


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
            self.d_activation = lambda x: (x > 0).astype(float)
        self.lr = lr
        self.W = xe_init((out_channels, in_channels, kernel_size, kernel_size))
        self.b = np.zeros((out_channels, 1))

    def forward(self, x):
        if self.padding > 0:
            x = np.pad(
                x,
                ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)),
                mode='constant'
            )

        self.x = x

        B, C, H, W = x.shape
        k = self.kernel_size
        s = self.stride
        H_out = (H - k) // s + 1
        W_out = (W - k) // s + 1

        cols = np.zeros((B, H_out * W_out, C * k * k))
        for b in range(B):
            col_idx = 0
            for i in range(0, H - k + 1, s):
                for j in range(0, W - k + 1, s):
                    window = x[b, :, i:i + k, j:j + k]  # (C, k, k)
                    cols[b, col_idx, :] = window.flatten()
                    col_idx += 1

        self.cols = cols
        self.H_out, self.W_out = H_out, W_out

        W_col = self.W.reshape(self.out_channels, -1)
        z = self.cols @ W_col.T + self.b.T  # (B, H_out*W_out, out_channels)
        z = z.transpose(0, 2, 1).reshape(B, self.out_channels, H_out, W_out)

        self.z = z
        return self.activation(z)

    def backward(self, delta):
        delta *= self.d_activation(self.z)
        B, F, H_out, W_out = delta.shape
        delta_flat = delta.reshape(B, F, -1)  # (B, F, H_out*W_out)

        W_rot = np.flip(self.W, axis=(2, 3))
        W_col = W_rot.reshape(F, -1)   # (F, C*k*k)

        dW = np.matmul(delta_flat, self.cols)  # (B, F, C*k*k)
        dW = dW.sum(axis=0).reshape(self.W.shape) / B  # (F, C*k*k)
        db = delta_flat.sum(axis=(0, 2)).reshape(self.b.shape) / B  # (F, 1)

        dcols = np.matmul(delta_flat.transpose(0, 2, 1), W_col)  # (B, N, C*k*k)
        dcols = dcols.reshape(B, H_out, W_out, self.in_channels, self.kernel_size, self.kernel_size)  # (B, H_out, W_out, C, k, k)
        dcols = dcols.transpose(0, 3, 1, 2, 4, 5)  # (B, C, H_out, W_out, k, k)

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
