import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def step_function(x):
    y = x > 0
    return y.astype(np.int32)


def sigmoid_function(x):
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


def plot_activate_functions():
    x = np.arange(-5.0, 5.0, .1)
    y1 = step_function(x)
    y2 = sigmoid_function(x)
    y3 = ReLU_function(x)
    plt.plot(x, y1)
    plt.plot(x, y2)
    plt.plot(x, y3)
    plt.show()


def init_network():
    network = {}
    network['W1'] = np.array([[.1, .3, .5], [.2, .4, .6]])
    network['b1'] = np.array([.1, .2, .3])
    network['W2'] = np.array([[.1, .4], [.2, .5], [.3, .6]])
    network['b2'] = np.array([.1, .2])
    network['W3'] = np.array([[.1, .3], [.2, .4]])
    network['b3'] = np.array([.1, .2])
    return network


def forward(network: dict, X: np.array) -> np.array:
    """
        A Simple Neural Network. X's shape is (2,);
    """
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    print("--- Layer 0 ---")
    print("X:", X, " Shape:", X.shape)

    print("--- layer 1 ---")

    a1 = np.dot(X, W1) + b1
    print("a1:", a1, "Shape:", a1.shape)
    z1 = sigmoid_function(a1)
    print("z1:", z1, " Shape:", z1.shape)

    print("--- layer 2 ---")

    a2 = np.dot(z1, W2) + b2
    print("a2:", a2, " Shape:", a2.shape)
    z2 = sigmoid_function(a2)
    print("z2:", z2, " Shape:", z2.shape)

    print("--- layer 3 ---")

    a3 = np.dot(z2, W3) + b3
    print("a3:", a3, " Shape:", a3.shape)
    Y = identity_function(a3)
    print("Y:", Y, " Shape:", Y.shape)

    return Y


def main():
    """
        main process
    """
    # plot_activate_functions()
    network = init_network()
    X = np.array([1.0, .5])
    Y = forward(network, X)


if __name__ == "__main__":
    main()
