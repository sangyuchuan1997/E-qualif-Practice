import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import os
from dataset.mnist import load_mnist
from PIL import Image
from functions import *


def img_show(img):
    pil_img = Image.fromarray(img)
    pil_img.show()


def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=True)
    return x_test, t_test


def init_network():
    with open("deep-learning/dataset/sample_weight.pkl", "rb") as f:
        network = pickle.load(f)
    return network


def predict(network, x):
    W1, W2, W3 = network["W1"], network["W2"], network["W3"]
    b1, b2, b3 = network["b1"], network["b2"], network["b3"]
    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)
    return y


def main():
    """
    main process
    """
    x, t = get_data()
    network = init_network()
    accuracy_cnt_single = 0
    for i in range(len(x)):
        y = predict(network, x[i])
        p = np.argmax(y)  # 最も確率の高い要素のインデックスを取得
        if p == t[i]:
            accuracy_cnt_single += 1
    print("Accuracy by single:", float(accuracy_cnt_single) / len(x))

    batch_size = 100
    accuracy_cnt_batch = 0
    for i in range(0, len(x), batch_size):
        x_batch = x[i : i + batch_size]
        y_batch = predict(network, x_batch)
        p = np.argmax(y_batch, axis=1)
        accuracy_cnt_batch += np.sum(p == t[i : i + batch_size])
    print("Accuracy by batch:", float(accuracy_cnt_batch) / len(x))


if __name__ == "__main__":
    main()
