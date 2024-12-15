import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import os
from dataset.mnist import load_mnist


def main():
    """
        main process
    """
    (x_train, t_train), (x_test, t_test) = load_mnist(
        flatten=True, normalize=False)
    print("x_train:", x_train, " Shape:", x_train.shape)
    print("t_train:", t_train, " Shape:", t_train.shape)
    print("x_test:", x_test, " Shape:", x_test.shape)
    print("t_test:", t_test, " Shape:", t_test.shape)


if __name__ == "__main__":
    main()
