import numpy as np
from dataset.mnist import load_mnist
from functions import *


class simpleNet:
    """クラスの説明"""

    def __init__(self):
        """初期化メソッド

        Args:
            param_description
        """
        # 初期化処理
        self.W = np.random.randn(2, 3)

    def predict(self, x):
        return np.dot(x, self.W)

    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error_onehot(y, t)

        return loss


def main():
    net = simpleNet()
    print(net.W)

    x = np.array([.6, .9])
    p = net.predict(x)
    print("p:", p, " Shape:", p.shape)
    print(np.argmax(p))
    t = np.array([0, 0, 1])
    print(net.loss(x, t))

    dW = numerical_gradient(lambda w: net.loss(x, t), net.W)
    print(dW)


if __name__ == "__main__":
    main()
