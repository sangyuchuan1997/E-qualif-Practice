from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from functions import *
from layers import *


class TwoLayerNet:
    """クラスの説明"""

    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        self.params = {}
        self.params['W1'] = weight_init_std * \
            np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * \
            np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

        # Layerの生成
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
        self.layers['Relu2'] = Relu()
        self.lastLayer = SoftmaxWithLoss()

    def predict(self, x):
        # # Old Way
        # W1, W2 = self.params['W1'], self.params['W2']
        # b1, b2 = self.params['b1'], self.params['b2']

        # a1 = np.dot(x, W1) + b1
        # z1 = sigmoid(a1)
        # a2 = np.dot(z1, W2) + b2
        # y = softmax(a2)

        # Use Layers
        for layer in self.layers.values():
            x = layer.forward(x)

        return x

    def loss(self, x, t):
        y = self.predict(x)
        return self.lastLayer.forward(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def numerical_gradient(self, x, t):
        def loss_W(W): return self.loss(x, t)
        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])

        return grads

    def gradient(self, x, t):
        # forward
        self.loss(x, t)
        
        # backward
        dout = 1
        dout = self.lastLayer.backword(dout)
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)
        
        grads = {}
        grads['W1'] = self.layers['Affine1'].dW
        grads['b1'] = self.layers['Affine1'].db
        grads['W2'] = self.layers['Affine2'].dW
        grads['b2'] = self.layers['Affine2'].db

        return grads

def main():
    """
        main process
    """
    # 勾配確認
    from dataset.mnist import load_mnist
    (x_train, t_train), (x_test, t_test) = load_mnist(
        normalize=True, one_hot_label=True)
    net = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)
    x_batch = x_train[:10]
    t_batch = t_train[:10]

    grads_numerical = net.numerical_gradient(x_batch, t_batch)
    grads_backprop = net.gradient(x_batch, t_batch)
    
    for key in grads_numerical.keys():
        diff = np.average(np.abs(grads_backprop[key] - grads_numerical[key]))
        print(key, ":", diff)


if __name__ == "__main__":
    main()
