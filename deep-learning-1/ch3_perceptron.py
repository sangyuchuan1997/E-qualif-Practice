import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def AND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.7
    return 0 if np.sum(x * w) + b <= 0 else 1


def NAND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])
    b = 0.7
    return 0 if np.sum(x * w) + b <= 0 else 1


def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.2
    return 0 if np.sum(x * w) + b <= 0 else 1


def XOR(x1, x2):
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    return AND(s1, s2)


def main():
    """
        main process
    """
    # perceptron パーセプトロン
    print(XOR(0, 0))
    print(XOR(0, 1))
    print(XOR(1, 1))


if __name__ == "__main__":
    main()
