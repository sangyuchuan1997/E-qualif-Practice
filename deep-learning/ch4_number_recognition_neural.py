import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from functions import *
from ch4_two_layer_net import TwoLayerNet
from dataset.mnist import load_mnist


def main():
    """
        main process
    """
    (x_train, t_train), (x_test, t_test) = load_mnist(
        normalize=True, one_hot_label=True)

    iters_num = 10000
    train_size = x_train.shape[0]
    batch_size = 100
    learning_rate = 0.001

    train_loss_list = []
    train_acc_list = []
    test_acc_list = []
    # 1エポック(epoch)あたりの繰り返し数
    iter_per_epoch = max(train_size / batch_size, 1)

    print("Network Initializing...")
    network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

    print("Initialized. Start training...")
    for i in range(iters_num):
        # ミニバッチの取得
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]

        # 勾配の計算
        # grad = network.numerical_gradient(x_batch, t_batch)
        grad = network.gradient(x_batch, t_batch)

        # パラメータの更新
        for key in ('W1', 'b1', 'W2', 'b2'):
            network.params[key] -= learning_rate * grad[key]

        # 学習経過の記録
        loss = network.loss(x_batch, t_batch)
        train_loss_list.append(loss)

        # 1エポックごとに認識精度を計算
        if i % iter_per_epoch == 0 or i == iters_num - 1:
            train_acc = network.accuracy(x_train, t_train)
            test_acc = network.accuracy(x_test, t_test)
            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)
            print("Iter#", i, "train acc, test acc |", train_acc, ",", test_acc)


if __name__ == "__main__":
    main()
