import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from functions import *


def plot_activate_functions():
    x = np.arange(-5.0, 5.0, 0.1)
    y1 = step_function(x)
    y2 = sigmoid(x)
    y3 = ReLU_function(x)
    plt.plot(x, y1)
    plt.plot(x, y2)
    plt.plot(x, y3)
    plt.show()


def init_network():
    network = {}
    network["W1"] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    network["b1"] = np.array([0.1, 0.2, 0.3])
    network["W2"] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    network["b2"] = np.array([0.1, 0.2])
    network["W3"] = np.array([[0.1, 0.3], [0.2, 0.4]])
    network["b3"] = np.array([0.1, 0.2])
    return network


def forward(network: dict, X: np.ndarray) -> np.array:
    """
    A Simple Neural Network. X's shape is (2,);
    """
    W1, W2, W3 = network["W1"], network["W2"], network["W3"]
    b1, b2, b3 = network["b1"], network["b2"], network["b3"]

    print("--- Layer 0 ---")
    print("X:", X, " Shape:", X.shape)

    print("--- layer 1 ---")

    a1 = np.dot(X, W1) + b1
    print("a1:", a1, "Shape:", a1.shape)
    z1 = sigmoid(a1)
    print("z1:", z1, " Shape:", z1.shape)

    print("--- layer 2 ---")

    a2 = np.dot(z1, W2) + b2
    print("a2:", a2, " Shape:", a2.shape)
    z2 = sigmoid(a2)
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
    X = np.array([1.0, 0.5])
    Y = forward(network, X)


if __name__ == "__main__":
    main()
