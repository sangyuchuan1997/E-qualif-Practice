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
    if np.all(np.isnan(a)):
        print("全てがNanになってしまいました.")
        return a
    a = a - np.nanmax(a)  # オーバーフロー対策
    exp_a = np.exp(a)
    sum_exp_a = np.sum(exp_a)
    return exp_a / sum_exp_a


def sum_squared_error(y, t):
    return 0.5 ** np.sum((y - t) ** 2)


def cross_entropy_error_simple(y, t):
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))


def cross_entropy_error_onehot(y: np.ndarray, t: np.ndarray):
    # ont hot 対応
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    batch_size = y.shape[0]
    return -np.sum(t * np.log(y + 1e-7)) / batch_size


def cross_entropy_error_label(y: np.ndarray, t: np.ndarray):
    # label 対応
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    batch_size = y.shape[0]
    return -np.sum(t * np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size


def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x

    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad

    return x


def numerical_gradient(f, x):
    h = 1e-4  # 0.0001
    grad = np.zeros_like(x)

    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = tmp_val + h
        fxh1 = f(x)  # f(x+h)

        x[idx] = tmp_val - h
        fxh2 = f(x)  # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2*h)

        x[idx] = tmp_val  # 値を元に戻す
        it.iternext()

    return grad
