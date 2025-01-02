import numpy as np
from dezero import utils
from dezero.core import Function, Variable, as_variable


# =============================================================================
# Basic functions: square / sin / cos / tanh / exp / log
# =============================================================================
class Square(Function):
    def forward(self, x) -> Variable:
        return x ** 2

    def backward(self, gy) -> Variable:
        x, = self.inputs
        gx = 2 * x * gy
        return gx


def square(x) -> Variable:
    return Square()(x)


class Exp(Function):
    def forward(self, x) -> Variable:
        return np.exp(x)

    def backward(self, gy) -> Variable:
        y = self.outputs[0]()
        gx = gy * y
        return gx


def exp(x) -> Variable:
    return Exp()(x)


class Sin(Function):
    def forward(self, x) -> Variable:
        y = np.sin(x)
        return y

    def backward(self, gy) -> Variable:
        x, = self.inputs
        gx = gy * cos(x)
        return gx


def sin(x) -> Variable:
    return Sin()(x)


class Cos(Function):
    def forward(self, x) -> Variable:
        y = np.cos(x)
        return y

    def backward(self, gy) -> Variable:
        x, = self.inputs
        gx = gy * -sin(x)
        return gx


def cos(x) -> Variable:
    return Cos()(x)


class Tanh(Function):
    def forward(self, x) -> Variable:
        y = np.tanh(x)
        return y

    def backward(self, gy) -> Variable:
        y = self.outputs[0]()
        gx = gy * (1 - y * y)
        return gx


def tanh(x) -> Variable:
    return Tanh()(x)


# =============================================================================
# Tensor operations: reshape / transpose / get_item / expand_dims / flatten
# =============================================================================
class Reshape(Function):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, x) -> Variable:
        self.x_shape = x.shape
        y = x.reshape(self.shape)
        return y

    def backward(self, gy) -> Variable:
        return reshape(gy, self.x_shape)


def reshape(x, shape) -> Variable:
    if x.shape == shape:
        return as_variable(x)
    return Reshape(shape)(x)


class Transpose(Function):
    def __init__(self, axes=None):
        self.axes = axes

    def forward(self, x) -> Variable:
        y = x.transpose(self.axes)
        return y

    def backward(self, gy) -> Variable:
        if self.axes is None:
            return transpose(gy)

        axes_len = len(self.axes)
        inv_axes = tuple(np.argsort([ax % axes_len for ax in self.axes]))
        return transpose(gy, inv_axes)


def transpose(x, axes=None) -> Variable:
    return Transpose(axes)(x)


class Sum(Function):
    def __init__(self, axis, keepdims):
        self.axis = axis
        self.keepdims = keepdims

    def forward(self, x) -> Variable:
        self.x_shape = x.shape
        y = x.sum(axis=self.axis, keepdims=self.keepdims)
        return y

    def backward(self, gy) -> Variable:
        gy = utils.reshape_sum_backward(
            gy, self.x_shape, self.axis, self.keepdims)
        gx = broadcast_to(gy, self.x_shape)
        return gx


def sum(x, axis=None, keepdims=False) -> Variable:
    return Sum(axis, keepdims)(x)


class GetItemGrad(Function):
    def __init__(self, slices, in_shape):
        self.slices = slices
        self.in_shape = in_shape

    def forward(self, gy) -> Variable:
        gx = np.zeros(self.in_shape)
        np.add.at(gx, self.slices, gy)
        return gx

    def backward(self, ggx) -> Variable:
        return get_item(ggx), self.slices


class GetItem(Function):
    def __init__(self, slices):
        self.slices = slices

    def forward(self, x) -> Variable:
        y = x[self.slices]
        return y

    def backward(self, gy) -> Variable:
        x, = self.inputs
        f = GetItemGrad(self.slices, x.shape)
        return f(gy)


def get_item(x, slices):
    return GetItem(slices)(x)


# =============================================================================
# sum / sum_to / broadcast_to / average / matmul / linear
# =============================================================================
class BroadcastTo(Function):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, x) -> Variable:
        self.x_shape = x.shape
        y = np.broadcast_to(x, self.shape)
        return y

    def backward(self, gy) -> Variable:
        gx = sum_to(gy, self.x_shape)
        return gx


def broadcast_to(x: Variable, shape) -> Variable:
    if x.shape == shape:
        return as_variable(x)
    return BroadcastTo(shape)(x)


class SumTo(Function):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, x) -> Variable:
        self.x_shape = x.shape
        y = utils.sum_to(x, self.shape)
        return y

    def backward(self, gy) -> Variable:
        gx = broadcast_to(gy, self.x_shape)
        return gx


def sum_to(x: Variable, shape) -> Variable:
    if x.shape == shape:
        return as_variable(x)
    return SumTo(shape)(x)


class MatMul(Function):
    def forward(self, x, W) -> Variable:
        y = x.dot(W)
        return y

    def backward(self, gy) -> Variable:
        x, W = self.inputs
        gx = matmul(gy, W.T)
        gW = matmul(x.T, gy)
        return gx, gW


def matmul(x, W) -> Variable:
    return MatMul()(x, W)


class Linear(Function):
    def forward(self, x, W, b) -> Variable:
        y = x.dot(W)
        if b is not None:
            y += b
        return y

    def backward(self, gy) -> Variable:
        x, W, b = self.inputs
        gb = None if b.data is None else sum_to(gy, b.shape)
        gx = matmul(gy, W.T)
        gW = matmul(x.T, gy)
        return gx, gW, gb


def linear(x, W, b=None) -> Variable:
    return Linear()(x, W, b)


def linear_simple(x, W, b=None):
    t = matmul(x, W)
    if b is None:
        return t

    y = t + b
    t.data = None
    return y


# =============================================================================
# activation function: sigmoid / relu / softmax / log_softmax / leaky_relu
# =============================================================================
def sigmoid_simple(x) -> Variable:
    x = as_variable(x)
    y = 1 / (1 + exp(-x))
    return y


class Sigmoid(Function):
    def forward(self, x) -> Variable:
        y = np.tanh(x * 0.5) * 0.5 + 0.5
        return y

    def backward(self, gy) -> Variable:
        y = self.outputs[0]()
        gx = gy * y * (1 - y)
        return gx


def sigmoid(x) -> Variable:
    return Sigmoid()(x)


class ReLU(Function):
    def forward(self, x) -> Variable:
        y = np.maximum(x, 0.0)
        return y

    def backward(self, gy) -> Variable:
        x, = self.inputs
        mask = x.data > 0
        gx = gy * mask
        return gx


def relu(x) -> Variable:
    return ReLU()(x)


class Softmax(Function):
    def __init__(self, axis=1):
        self.axis = axis

    def forward(self, x) -> Variable:
        y = x - np.max(axis=self.axis, keepdims=True)  # Overflow対応
        y = np.exp(y)
        y /= y.sum(axis=self.axis, keepdims=True)

    def backward(self, gy) -> Variable:
        y = self.outputs[0]()
        gx = y * gy
        sumdx = gx.sum(axis=self.axis, keepdims=True)
        gx -= y * sumdx
        return gx


def softmax(x, axis=1) -> Variable:
    return Softmax(axis=axis)(x)


# =============================================================================
# loss function: mean_squared_error / softmax_cross_entropy / sigmoid_cross_entropy / binary_cross_entropy
# =============================================================================
class MeanSquaredError(Function):
    def forward(self, x0, x1) -> Variable:
        diff = x0 - x1
        y = (diff ** 2).sum() / len(diff)
        return y

    def backward(self, gy) -> Variable:
        x0, x1 = self.inputs
        diff = x0 - x1
        gy = broadcast_to(gy, diff.shape)
        gx0 = gy * diff * (2. / len(diff))
        gx1 = -gx0
        return gx0, gx1


def mean_squared_error(x0, x1) -> Variable:
    return MeanSquaredError()(x0, x1)


class SoftmaxCrossEntropy(Function):
    def forward(self, x, t) -> Variable:
        N = x.shape[0]
        log_z = utils.logsumexp(x, axis=1)
        log_p = x - log_z
        log_p = log_p[np.arange[N], t.ravel()]
        y = -log_p.sum() / np.float32(N)
        return y

    def backward(self, gy) -> Variable:
        x, t = self.inputs
        N, CLS_NUM = x.shape

        gy *= 1/N
        y = softmax(x)

        t_onehot = np.eye(CLS_NUM, dtype=t.dtype)[t.data]
        y = (y - t_onehot) * gy


def softmax_cross_entropy(x, t):
    return SoftmaxCrossEntropy()(x, t)
