import numpy as np
from functions import *


class Relu:

    def __init__(self):
        """初期化処理"""
        self.mask = None

    def forward(self, x: np.ndarray):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0

        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout

        return dx


class Sigmoid:

    def __init__(self):
        """初期化処理"""
        self.out = None

    def forward(self, x: np.ndarray):
        out = 1 / (1 + np.exp(-x))
        self.out = out

        return out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out

        return dx


class Affine:

    def __init__(self, W: np.ndarray, b: np.ndarray):
        """初期化処理"""
        self.W = W
        self.b = b
        self.x = None
        self.original_x_shape = None
        self.dW = None
        self.db = None

    def forward(self, x: np.ndarray):
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        self.x = x

        out = np.dot(self.x, self.W) + self.b

        return out

    def backward(self, dout: np.ndarray):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)

        dx = dx.reshape(*self.original_x_shape)
        return dx


class SoftmaxWithLoss:

    def __init__(self):
        """初期化処理"""
        self.loss = None
        self.y = None
        self.t = None

    def forward(self, x: np.ndarray, t: np.ndarray):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error_onehot(self.y, self.t)

        return self.loss

    def backword(self, dout=1):
        batch_size = self.t.shape[0]
        if self.t.size == self.y.size:  # 教師データがone-hot-vectorの場合
            dx = (self.y - self.t) / batch_size
        else:
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t] -= 1
            dx = dx / batch_size

        return dx
