from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from linear_regression_gradient_decent import LinearRegressionUsingGD


def plot_result(x, y, y_predicted):
    plt.subplots(1, 1)

    # Actual data points
    plt.scatter(x, y, s=10)
    plt.xlabel('\u03B80')
    plt.ylabel('\u03B81')

    # Predicted data points
    plt.plot(x, y_predicted, color='r')
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
    ax.set_zlim(0, 25)
    plt.show()


def get_cost_function(theta_1, theta_2, x, y, points_n):
    m = theta_1.shape[0]
    cost = np.zeros([theta_1.shape[0], theta_1.shape[1]])
    for i in range(points_n):
        residuals = ((theta_1 * x[i] + theta_2) - y[i]) ** 2
        cost += residuals
    cost = cost / (2 * m)
    return cost


def plot_cost_function(theta_1, theta_2, x, y, points_n):
    cost = get_cost_function(theta_1, theta_2, x, y, points_n)
    plot_cost_function_3d(theta_1, theta_2, cost)
    plot_cost_function_2d(theta_1, theta_2, cost)


def plot_raw_data(x, y):
    plt.scatter(x, y, s=10)
    plt.xlabel('\u03B80')
    plt.ylabel('\u03B81')
    plt.show()


def create_mesh_grid():
    theta_1 = np.arange(-10, 14, 0.05)
    theta_2 = np.arange(-100, 100, 0.05)
    theta_1, theta_2 = np.meshgrid(theta_1, theta_2)
    return theta_1, theta_2


def create_data(points_n):
    x = np.random.rand(points_n, 1) * 20
    y = (2 * (x + (2 * np.random.rand(points_n, 1)))) + 1
    return x, y


def main():
    points_n = 50
    x, y = create_data(points_n)
    theta_1, theta_2, = create_mesh_grid()

    plot_raw_data(x, y)
    plot_cost_function(theta_1, theta_2, x, y, points_n)

    # Model initialization
    # Sci-kit learn implementation:
    # regression_model = LinearRegression()
    regression_model = LinearRegressionUsingGD()

    # Fit the data(train the model)
    regression_model.fit(x, y)

    # Predict
    y_predicted = regression_model.predict(x)

    # model evaluation
    rmse = mean_squared_error(y, y_predicted)
    r2 = r2_score(y, y_predicted)

    # For sci-kit learn implementation:
    # print('Slope:', regression_model.coef_)
    # print('Intercept:', regression_model.intercept_)
    print('Slope:', regression_model.w)
    print('Intercept:', regression_model.b)
    print('Root mean squared error: ', rmse)
    print('R2 score: ', r2)

    # plot
    plot_result(x, y, y_predicted)


if __name__ == '__main__':
    main()
