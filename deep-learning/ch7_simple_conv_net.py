from collections import OrderedDict
from layers import *
from functions import *
from optimizer import *
from dataset.mnist import load_mnist


class SimpleConvNet:

    def __init__(self, input_dim=(1, 28, 28), conv_param={'filter_num': 30, 'filter_size': 5, 'pad': 0, 'stride': 1}, hidden_size=100, output_size=10, weight_init_std=0.01):
        """初期化処理"""
        filter_num = conv_param['filter_num']
        filter_size = conv_param['filter_size']
        filter_pad = conv_param['pad']
        filter_stride = conv_param['stride']
        input_size = input_dim[1]
        conv_output_size = (input_size - filter_size + 2 *
                            filter_pad) / filter_stride + 1
        pool_output_size = int(
            filter_num * (conv_output_size/2) * (conv_output_size/2))

        self.params = {}
        self.params['W1'] = weight_init_std * \
            np.random.randn(filter_num, input_dim[0], filter_size, filter_size)
        self.params['b1'] = np.zeros(filter_num)
        self.params['W2'] = weight_init_std * \
            np.random.randn(pool_output_size, hidden_size)
        self.params['b2'] = np.zeros(hidden_size)
        self.params['W3'] = weight_init_std * \
            np.random.randn(hidden_size, output_size)
        self.params['b3'] = np.zeros(output_size)

        self.layers = OrderedDict()
        self.layers['Conv1'] = Convolution(
            self.params['W1'], self.params['b1'], conv_param['stride'], conv_param['pad'])
        self.layers['Relu1'] = Relu()
        self.layers['Pool1'] = Pooling(pool_h=2, pool_w=2, stride=2)
        self.layers['Affine1'] = Affine(self.params['W2'], self.params['b2'])
        self.layers['Relu2'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W3'], self.params['b3'])

        self.last_layer = SoftmaxWithLoss()

    def predict(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    def loss(self, x, t):
        y = self.predict(x)
        return self.last_layer.forward(y, t)

    def gradient(self, x, t):
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # settings
        grads = {}
        grads['W1'] = self.layers['Conv1'].dW
        grads['b1'] = self.layers['Conv1'].db
        grads['W2'] = self.layers['Affine1'].dW
        grads['b2'] = self.layers['Affine1'].db
        grads['W3'] = self.layers['Affine2'].dW
        grads['b3'] = self.layers['Affine2'].db

        return grads

    def accuracy(self, x, t, batch_size=100):
        if t.ndim != 1:
            t = np.argmax(t, axis=1)

        acc = 0.0

        for i in range(int(x.shape[0] / batch_size)):
            tx = x[i*batch_size:(i+1)*batch_size]
            tt = t[i*batch_size:(i+1)*batch_size]
            y = self.predict(tx)
            y = np.argmax(y, axis=1)
            acc += np.sum(y == tt)

        return acc / x.shape[0]


def main():
    """
        main process
    """
    (x_train, t_train), (x_test, t_test) = load_mnist(
        normalize=True, one_hot_label=True, flatten=False)

    epochs = 20
    iters_num = 10000
    train_size = x_train.shape[0]
    batch_size = 100
    learning_rate = 0.001

    train_loss_list = []
    train_acc_list = []
    test_acc_list = []
    # 1エポック(epoch)あたりの繰り返し数
    iter_per_epoch = max(train_size / batch_size, 1)
    max_iter = int(epochs * iter_per_epoch)
    current_iter = 0
    current_epoch = 0

    print("Network Initializing...")
    network = SimpleConvNet()
    optimizer = Adam(lr=learning_rate)
    # optimizer = SGD(lr=learning_rate)

    print("Initialized. Start training...")
    for i in range(max_iter):
        # ミニバッチの取得
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]

        # 勾配の計算
        grads = network.gradient(x_batch, t_batch)

        # パラメータの更新
        optimizer.update(network.params, grads)

        # 学習経過の記録
        loss = network.loss(x_batch, t_batch)
        train_loss_list.append(loss)
        # print("train loss:", loss)

        # 1エポックごとに認識精度を計算
        if current_iter % iter_per_epoch == 0:
            current_epoch += 1
            train_acc = network.accuracy(x_train[:1000], t_train[:1000])
            test_acc = network.accuracy(x_test[:1000], t_test[:1000])
            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)
            print("=== epoch:", current_epoch, "train acc, test acc |",
                  train_acc, ",", test_acc)
        current_iter += 1


if __name__ == "__main__":
    main()
