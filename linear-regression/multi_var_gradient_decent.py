import matplotlib.pyplot as plt
import numpy as np


class LinearRegressionUsingGD:
    """Linear Regression Using Gradient Descent.

    Parameters
    ----------
    lr : float
        Learning rate
    n_iterations : int
        Number of gradient decent iterations

    Attributes
    ----------
    w : weights/ after fitting the model
    cost_ : total error of the model after each iteration

      Public Methods
    -------
    fit(x, y)
        Fit model h(x) to the training data
    predict(x)
        Predict Y given the training data X, using the trained model h(x)
    """

    def __init__(self, lr, n_iterations):
        self.lr = lr
        self.n_iterations = n_iterations
        self.cost_ = []
        self.w = []

    def fit(self, x, y):
        """Fit the training data
        Parameters
        ----------
        x : array-like, shape = [n_samples, n_features]
            Training samples
        y : array-like, shape = [n_samples, n_target_values]
            Target values
        Returns
        -------
        self : object
        """

        self.cost_ = []
        self.w = np.zeros((x.shape[1]+1, 1))

        bias = np.ones(shape=(len(x), 1))
        x = np.append(bias, x, axis=1)

        n = x.shape[0]

        for i in range(self.n_iterations):
            y_pred = np.dot(x, self.w)
            residuals = y_pred - y
            gradient_vector = np.dot(x.T, residuals)
            self.w -= (self.lr * gradient_vector) / n
            cost = np.sum((residuals ** 2)) / (2 * n)
            self.cost_.append(cost)
            # Log Progress
            if i % 10 == 0:
                print(
                    "iter={},    weight={}.    cost={}".format(i, self.w, cost)
                )

        plt.subplots(1, 1)
        plt.plot(range(0, self.n_iterations), self.cost_)
        plt.xlabel('Iterations')
        plt.ylabel('J(\u03B80, \u03B81)')
        plt.show()

        return self

    def predict(self, x):
        """ Predicts the value after the model has been trained.
        Parameters
        ----------
        x : array-like, shape = [n_samples, n_features]
            Test samples
        Returns
        -------
        Predicted value
        """
        bias = np.ones(shape=(len(x), 1))
        x = np.append(bias, x, axis=1)
        return np.dot(x, self.w)
