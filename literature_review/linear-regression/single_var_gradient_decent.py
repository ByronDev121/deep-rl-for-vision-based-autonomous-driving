import matplotlib.pyplot as plt
import numpy as np


class LinearRegressionUsingGD:
    """
    A class used to represent Gradient Decent for Linear Regression
    ...

    Attributes
    ----------
    n_iterations : int
        number of gradient decent iterations
    lr : float
        learning rate  - make sure to set this small e.g. < 0.001
    w : float
        weights variable, in this case w is the gradient tern in y = mx +c ---> y = wc + b
    b : float
        bias variable, in this case w is the gradient tern in y = mx +c ---> y = wc + b

    Public Methods
    -------
    fit(x, y)
        Fit model h(x) to the training data
    predict(x)
        Predict Y given the training data X, using the trained model h(x)
    """

    def __init__(self, lr=0.0005, n_iterations=100):
        self.n_iterations = n_iterations
        self.lr = lr
        self.w = np.random.rand(1)
        self.b = np.random.rand(1)

    def _cost_function(self, x: object, y: object) -> object:
        total = len(x)
        total_error = 0.0
        for i in range(total):
            total_error += (y[i] - (self.w * x[i] + self.b)) ** 2
        return total_error / total

    def _update_weights(self, x: object, y: object):
        weight_deriv = 0
        bias_deriv = 0
        length = len(x)
        for i in range(length):
            # Calculate partial derivatives

            # -2x(y - (mx + b))
            weight_deriv += -2 * x[i] * (y[i] - (self.w * x[i] + self.b))

            # -2(y - (mx + b))
            bias_deriv += -2 * (y[i] - (self.w * x[i] + self.b))

        # We subtract because the derivatives point in direction of steepest ascent
        self.w -= (weight_deriv / length) * self.lr
        self.b -= (bias_deriv / length) * self.lr

    def fit(self, x, y):
        """Fit model h(x) to the training data
        Parameters
        ----------
        x : array-like, shape = [n_samples, n_features]
            Training samples
        y : array-like, shape = [n_samples, n_target_values]
            Target values
        """
        cost_x = []
        cost_y = []

        for i in range(self.n_iterations):
            self._update_weights(x, y)

            # Calculate cost for auditing purposes
            cost = self._cost_function(x, y)
            cost_x.append(i)
            cost_y.append(cost)

            # Log Progress
            if i % 10 == 0:
                print(
                    "iter={},    weight={}    bias={}    cost={}".format(i, self.w, self.b, cost)
                )

        plt.subplots(1, 1)
        plt.plot(cost_x, cost_y)
        plt.xlabel('Iterations')
        plt.ylabel('J(\u03B80, \u03B81)')
        plt.show()

    def predict(self, x):
        """Predict Y given the training data X, using the trained model h(x)
        Parameters
        ----------
        x : array-like, shape = [n_samples, n_features]
          Training samples

        Returns
        ----------
        y : array-like, shape = [n_samples, n_target_values]
            Target values
        """
        return self.w * x + self.b
