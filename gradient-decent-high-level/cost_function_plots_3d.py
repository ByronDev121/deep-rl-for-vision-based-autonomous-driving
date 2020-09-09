from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np


def plot_cost_function_2d(theta_1, theta_2, cost):
    fig, ax = plt.subplots(1, 1)
    ax.contourf(theta_1,
                theta_2,
                cost,
                # levels=[0, 1, 2, 4, 6, 8, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                cmap=cm.coolwarm,
                antialiased=True)

    plt.xlabel('\u03B80')
    plt.ylabel('\u03B81')
    plt.show()


def plot_cost_function_3d(theta_1, theta_2, cost):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(theta_1,
                    theta_2,
                    cost,
                    cmap=cm.coolwarm,
                    linewidth=0, antialiased=True, )

    plt.xlabel('\u03B80')
    plt.ylabel('\u03B81')
    ax.set_zlabel('J(\u03B80, \u03B81)')
    # ax.set_zlim(0, 25)
    plt.show()


def plot_cost_function(theta_1, theta_2):
    cost = theta_1 ** 2 + theta_2 ** 2
    plot_cost_function_3d(theta_1, theta_2, cost)
    plot_cost_function_2d(theta_1, theta_2, cost)


def create_mesh_grid():
    theta_1 = np.arange(-10, 10, 0.05)
    theta_2 = np.arange(-10, 10, 0.05)
    theta_1, theta_2 = np.meshgrid(theta_1, theta_2)
    return theta_1, theta_2


def main():
    theta_1, theta_2 = create_mesh_grid()
    plot_cost_function(theta_1, theta_2)


if __name__ == '__main__':
    main()