import numpy as np
from scipy.optimize import fmin_tnc


class GradientDescent:
    """Linear Regression Using Gradient Descent.

        Parameters
        ----------

        Public Methods
        -------
        fit(x, y)
            Fit model h(x) to the training data
        predict(x)
            Predict Y given the training data X, using the trained model h(x)
    """

    def __init__(self):
        self.theta = []

    @staticmethod
    def sigmoid(z):
        # Activation function used to map any real value between 0 and 1
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def net_input(x, theta):
        # Computes the weighted sum of inputs
        return np.dot(x, theta)

    def probability(self, x, theta):
        # Returns the probability after passing through sigmoid
        return self.sigmoid(self.net_input(x, theta))

    def cost_function(self, theta, x, y):
        # Computes the cost function for all the training samples
        m = x.shape[0]
        total_cost = -(1 / m) * np.sum(
            y * np.log(self.probability(x, theta)) + (1 - y) * np.log(
                1 - self.probability(x, theta)))
        return total_cost

    def gradient(self, theta, x, y):
        # Computes the gradient of the cost function at the point theta
        m = x.shape[0]
        return (1 / m) * np.dot(x.T, self.sigmoid(self.net_input(x, theta)) - y)

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

        theta = np.zeros((x.shape[1], 1))

        opt_weights = fmin_tnc(
            func=self.cost_function,
            x0=theta,
            fprime=self.gradient,
            args=(x, y)
        )

        self.theta = opt_weights[0]

        return self

    def predict(self, x):
        theta = self.theta[:, np.newaxis]
        return self.probability(x, theta)
