import numpy as np

from avgpool2d import AvgPool2D
from conv2d import Conv2D
from dense import Dense
from flatten import Flatten
from lenet5 import LeNet5
from utils import load_data, precision, recall, f1_score, accuracy

LR = 0.1
EPOCHS = 10
BATCH_SIZE = 64

LAYERS = [
            Conv2D(6, 1, 5, activation=np.tanh, lr=LR),
            AvgPool2D(2, 2),
            Conv2D(16, 6, 5, activation=np.tanh, lr=LR),
            AvgPool2D(2, 2),
            Flatten(),
            Dense(16*4*4, 120, activation=np.tanh, lr=LR),
            Dense(120, 84, activation=np.tanh, lr=LR),
            Dense(84, 10, activation=None, lr=LR)
        ]

if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = load_data('mnist.npz')

    x_train = x_train[:, np.newaxis, :, :]  # (N, 1, 28, 28)
    x_test = x_test[:, np.newaxis, :, :]

    model = LeNet5(LAYERS)
    losses = model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE)
    y_pred = model.predict(x_test)

    accuracy = accuracy(y_test, y_pred)
    prec = precision(y_test, y_pred)
    rec = recall(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f'Accuracy на тестовых примерах: {accuracy * 100:.2f}%')
    print(f'Precision: {prec:.4f}')
    print(f'Recall: {rec:.4f}')
    print(f'F1 Score: {f1:.4f}')

