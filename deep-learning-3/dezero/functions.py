import numpy as np
from dezero.core import Function, Variable, as_variable


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
        return x ** 2

    def backward(self, gy) -> Variable:
        x, = self.inputs
        gx = np.exp(x) * gy
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


class Reshape(Function):
    def __init__(self, shape) -> None:
        self.shape = shape

    def forward(self, x: Variable | np.ndarray) -> Variable:
        self.x_shape = x.shape
        y = x.reshape(self.shape)
        return y

    def backward(self, gy) -> Variable:
        return reshape(gy, self.x_shape)


def reshape(x: Variable, shape):
    if x.shape == shape:
        return as_variable(x)
    return Reshape(shape)(x)


class Transpose(Function):
    def __init__(self, axes=None) -> None:
        self.axes = axes

    def forward(self, x: Variable | np.ndarray) -> Variable:
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
