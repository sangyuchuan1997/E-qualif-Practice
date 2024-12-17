import numpy as np

def step_function(x):
    y = x > 0
    return y.astype(np.int32)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def ReLU_function(x):
    return np.maximum(0, x)


def identity_function(x):
    return x


def softmax(a):
    a = a - np.max(a)  # オーバーフロー対策
    exp_a = np.exp(a)
    sum_exp_a = np.sum(exp_a)
    return exp_a / sum_exp_a


def sum_squared_error(y, t):
    return 0.5 ** np.sum((y - t) ** 2)


def cross_entropy_error_simple(y, t):
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))


def cross_entropy_error_onehot(y : np.ndarray, t : np.ndarray):
    # ont hot 対応
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    batch_size = y.shape[0]
    return -np.sum(t * np.log(y + 1e-7)) / batch_size


def cross_entropy_error_label(y : np.ndarray, t : np.ndarray):
    # label 対応
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    batch_size = y.shape[0]
    return -np.sum(t * np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size