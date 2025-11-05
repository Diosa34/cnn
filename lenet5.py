import numpy as np
from tqdm import tqdm

from utils import cross_entropy_loss, d_cross_entropy_loss, to_one_hot


class LeNet5:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def fit(self, x, y, epochs=1, batch_size=64):
        x = x.astype(np.float32) / 255.0
        y_onehot = to_one_hot(y, 10)
        losses = []
        for ep in range(epochs):
            perm = np.random.permutation(x.shape[0])
            x, y_onehot = x[perm], y_onehot[perm]
            batch_losses = []
            for i in tqdm(range(0, x.shape[0], batch_size)):
                xb, yb = x[i:i+batch_size], y_onehot[i:i+batch_size]
                logits = self.forward(xb)
                loss = cross_entropy_loss(logits, yb)
                batch_losses.append(loss)
                grad = d_cross_entropy_loss(logits, yb)
                self.backward(grad)
            epoch_loss = np.mean(batch_losses)
            losses.append(epoch_loss)
            print("epoch_loss", epoch_loss)
        return losses

    def predict(self, x):
        logits = self.forward(x)
        return np.argmax(logits, axis=1)
