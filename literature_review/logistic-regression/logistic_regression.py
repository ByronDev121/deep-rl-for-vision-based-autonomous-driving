import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from gradient_descent import GradientDescent


def sigmoid(x):
    a = []
    for item in x:
        a.append(1/(1+math.exp(-item)))
    return a


def plot_sigmoid():
    x = np.arange(-10, 10, 0.2)
    sig = sigmoid(x)
    fig, ax = plt.subplots()
    ax.plot(x, sig, label='g(z) = 1 / 1 - x^-z')
    plt.legend()
    ax.set(xlabel='z', ylabel='g(z)')
    ax.grid()

    plt.show()


def log(x):
    a = []
    b = []
    for item in x:
        a.append(-math.log(item))
        b.append(-math.log(1-item))
    return a, b


def plot_log():
    x = np.arange(0.01, 1, 0.01)
    f1, f2 = log(x)
    fig, ax = plt.subplots()
    ax.plot(x, f1, label='J(\u03B8) = - log( h(x) )')
    ax.plot(x, f2, label='J(\u03B8) = - log( 1 - h(x) )')
    plt.legend()
    ax.set(xlabel='h(x)', ylabel='J(\u03B8)')
    ax.grid()

    plt.show()


def load_data(path, header):
    df = pd.read_csv(path, header=header)
    return df


def accuracy(y_predicted, actual_classes, probab_threshold=0.5):
    predicted_classes = ( y_predicted >=
                         probab_threshold).astype(int)
    predicted_classes = predicted_classes.flatten()
    accuracy = np.mean(predicted_classes == actual_classes)
    return accuracy * 100


def raw_plot(malignant, benign):
    plt.scatter(malignant.iloc[:, 1], malignant.iloc[:, 2], s=10, label='Malignant')
    plt.scatter(benign.iloc[:, 1], benign.iloc[:, 2], s=10, label='Benign')
    plt.legend()
    plt.xlabel('Mean Area')
    plt.ylabel('Mean Compactness')
    plt.show()


def plot_decision_boundary(parameters, malignant, benign):
    x_values = [0.1, 0.4]
    y_values = - (parameters[0] + np.dot(parameters[1], x_values)) / parameters[2]

    plt.scatter(malignant.iloc[:, 1], malignant.iloc[:, 2], s=10, label='Malignant')
    plt.scatter(benign.iloc[:, 1], benign.iloc[:, 2], s=10, label='Benign')
    plt.plot(x_values, y_values, label='Decision Boundary')
    plt.xlabel('Mean Area')
    plt.ylabel('Mean Compactness')
    plt.legend()
    plt.show()


def get_data_frame():
    df = pd.read_csv('breast cancer.csv',
                     delimiter=',',
                     names=["id", "diagnosis", "radius_mean", "texture_mean", "perimeter_mean",
                            "area_mean", "smoothness_mean", "compactness_mean", "concavity_mean",
                            "concave points_mean", "symmetry_mean", "fractal_dimension_mean", "radius_se",
                            "texture_se", "perimeter_se", "area_se", "smoothness_se", "compactness_se",
                            "concavity_se", "concave points_se", "symmetry_se", "fractal_dimension_se",
                            "radius_worst", "texture_worst", "perimeter_worst", "area_worst", "smoothness_worst",
                            "compactness_worst", "concavity_worst", "concave points_worst", "symmetry_worst",
                            "fractal_dimension_worst", ])

    df["diagnosis"] = df["diagnosis"].replace('M', 1)
    df["diagnosis"] = df["diagnosis"].replace('B', 0)

    df.insert(1, 'Theta 0', 1)
    X = df.iloc[:, [1, 6, 7]]
    # X = df.iloc[:, :]

    X['area_mean'] = X['area_mean'] / np.max(X['area_mean'])
    X['smoothness_mean'] = X['smoothness_mean'] / np.max(X['smoothness_mean'])

    # y = target values, last column of the data frame
    y = df.iloc[:, 2]

    # filter out the applicants that got admitted
    malignant = X.loc[y == 1]

    # filter out the applicants that din't get admission
    benign = X.loc[y == 0]

    return X, y, malignant, benign


if __name__ == "__main__":
    plot_sigmoid()
    plot_log()

    X, y, malignant, benign = get_data_frame()

    raw_plot(malignant, benign)

    regression_model = GradientDescent()
    model = regression_model.fit(X, y)
    y_predicted = regression_model.predict(X)

    plot_decision_boundary(model.theta, malignant, benign)

    print('Parameters', model.theta)
    print('Accuracy', accuracy(y_predicted, y))
