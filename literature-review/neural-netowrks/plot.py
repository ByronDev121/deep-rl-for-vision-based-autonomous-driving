import matplotlib.pyplot as plt
import numpy as np


def raw_plot(y1, y2, x3, y3):
    plt.scatter(y1[0], y1[1], s=10, label='0')
    plt.scatter(y2[0], y2[1], s=10, label='1')
    plt.plot(x3, y3, color='green', label='decision boundary')
    plt.legend()
    plt.grid(alpha=.4, linestyle='--')
    # plt.axhline(y=0, xmin=0, xmax=1, color='black', alpha=.4)
    # plt.axvline(x=0, ymin=0, ymax=1, color='black', alpha=.4)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()


def f(x):
    return -x + 1.5


if __name__ == "__main__":
    t1 = np.arange(0.0, 5.0, 0.1)

    y1 = ([1], [1])
    y2 = ([0, 0, 1], [0, 1, 0])

    x = np.arange(0.0, 1.8, 0.5)

    raw_plot(y1, y2, x, f(x))
