import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dataset.mnist import load_mnist


def main():
    """
        main process
    """
    (x_train, t_train), (x_test, t_test) = load_mnist(one_hot_label=True)


if __name__ == "__main__":
    main()
