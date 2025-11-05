import numpy as np


def load_data(path):
    with np.load(path) as f:
        x_train, y_train = f['x_train'], f['y_train']
        x_test, y_test = f['x_test'], f['y_test']
        return (x_train, y_train), (x_test, y_test)


# === Активации ===
def relu(x):
    return np.maximum(0, x)


def d_relu(x):
    return (x > 0).astype(float)


def tanh(x):
    return np.tanh(x)


def d_tanh(x):
    return 1 - np.tanh(x) ** 2


# === Softmax и Cross-Entropy ===
def softmax(x):
    e = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def cross_entropy_loss(logits, y_true):
    probs = softmax(logits)
    return -np.mean(np.sum(y_true * np.log(probs + 1e-12), axis=1))


def d_cross_entropy_loss(logits, y_true):
    probs = softmax(logits)
    return (probs - y_true) / logits.shape[0]


def to_one_hot(y, num_classes):
    oh = np.zeros((y.shape[0], num_classes))
    oh[np.arange(y.shape[0]), y] = 1
    return oh


def xe_init(shape):
    fan_in = np.prod(shape[1:])
    return np.random.randn(*shape) * np.sqrt(2. / fan_in)


def accuracy(y_true, y_pred):
    return np.mean(y_pred == y_true)


def precision(y_true, y_pred, average='macro'):
    classes = np.unique(y_true)
    precisions = []
    for c in classes:
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        precisions.append(p)

    return _metric_with_average(precisions, y_true, average)


def recall(y_true, y_pred, average='macro'):
    classes = np.unique(y_true)
    recalls = []
    for c in classes:
        tp = np.sum((y_true == c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        recalls.append(r)

    return _metric_with_average(recalls, y_true, average)


def f1_score(y_true, y_pred, average='macro'):
    precisions = precision(y_true, y_pred, average=None)
    recalls = recall(y_true, y_pred, average=None)
    f1s = []
    for p, r in zip(precisions, recalls):
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0
        f1s.append(f)

    return _metric_with_average(f1s, y_true, average)


def _metric_with_average(metric, y_true, average):
    if average == 'macro':
        return np.mean(metric)
    elif average == 'weighted':
        counts = [np.sum(y_true == c) for c in np.unique(y_true)]
        return np.average(metric, weights=counts)
    else:
        return metric
