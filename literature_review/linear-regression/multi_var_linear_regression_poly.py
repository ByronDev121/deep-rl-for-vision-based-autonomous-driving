import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from multi_var_gradient_decent import LinearRegressionUsingGD
from mpl_toolkits.mplot3d import axes3d
from sklearn.metrics import mean_squared_error, r2_score

def create_mesh_grid(start, end):
    theta_1 = np.linspace(start, end, 30)
    theta_2 = np.linspace(start, end, 30)
    theta_1, theta_2 = np.meshgrid(theta_1, theta_2)
    return theta_1, theta_2

def plot_result(x, y, y_pred):
    fig = plt.figure()
    plt.scatter(x, y)
    plt.scatter(x, y_pred, color='red')
    plt.xlabel('\u03B80')
    plt.ylabel('\u03B81')
    plt.show()


def plot_cost_function_2d(theta_1, theta_2, cost):
    fig, ax = plt.subplots(1, 1)
    ax.contourf(theta_1,
                theta_2,
                cost,
                levels=[0, 1, 2, 4, 6, 8, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                cmap=cm.coolwarm,
                antialiased=True)

    plt.xlabel('\u03B81')
    plt.ylabel('\u03B82')
    plt.show()


def get_cost_function(theta_1, theta_2, x1, y, points_n):
    m = theta_1.shape[0]
    cost = np.zeros([theta_1.shape[0], theta_1.shape[1]])
    for i in range(points_n):
        residuals = ((theta_1 * x1[i] + theta_2) - y[i]) ** 2
        cost += residuals
    cost = cost / (2 * m)
    return cost


def plot_cost_function(x, y, points_n):
    theta_1, theta_2, = create_mesh_grid(0, 30)
    cost = get_cost_function(theta_1, theta_2, x, y, points_n)
    plot_cost_function_2d(theta_1, theta_2, cost)


def plot_raw_data(x, y):
    fig = plt.figure()
    plt.scatter(x,y)
    plt.xlabel('\u03B80')
    plt.ylabel('\u03B81')
    plt.show()


def create_data(points_n):
    np.random.seed(0)
    x = np.random.rand(points_n, 2)
    x1 = x[:, 0].reshape(points_n, 1)
    x2 = x[:, 0].reshape(points_n, 1) ** (8)

    x[:, 0] = x1.reshape(points_n,)
    x[:, 1] = x2.reshape(points_n,)

    x1 = x1 / x1.max()
    x2 = x2 / x2.max()

    y = 2 + 3 * x1 + np.random.rand(points_n, 1) + 32 * x2 + np.random.rand(points_n, 1)

    return x, x1, x2, y


def main():
    points_n = 100
    x, x1, x2, y = create_data(points_n)

    plot_raw_data(x1, y)
    plot_cost_function(x1, y, points_n)
    plot_cost_function(x2, y, points_n)

    # Model initialization
    regression_model = LinearRegressionUsingGD(lr=0.05, n_iterations=10000)
    # Fit the data(train the model)
    regression_model.fit(x, y)
    # Predict
    y_predicted = regression_model.predict(x)

    # model evaluation
    rmse = mean_squared_error(y, y_predicted)
    r2 = r2_score(y, y_predicted)

    # For sci-kit learn implementation:
    print('Weights:', regression_model.w)
    print('Root mean squared error: ', rmse)
    print('R2 score: ', r2)

    # plot
    plot_result(x1, y, y_predicted)


if __name__ == '__main__':
    main()


