"""
Write full code for CustomLinearRegression Class
"""
import numpy as np
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression as lg_sklearn
from sklearn.linear_model import Ridge

from typing import Union


class CustomLinearRegression:
    def __init__(self, C: Union[float, int] = 0, random_state: int = 42):
        self.random_state = random_state
        self.C = C  # L2 regularization coefficient, you will need this in gradient computation
        self.W = None
        self.b = None

    def init_weights(self, input_size: int, output_size: int):
        """
        Initialize weights and biases

        `W` -  matrix with shapes (input_size, output_size)
        Initialize with random numbers from Normal distribution with mean 0 and std 0.01
        `b` - vector with shape (1, output_size), initialize with 0s
        """
        np.random.seed(self.random_state)
        self.W = ...  # TODO
        self.b = ...  # TODO

    def fit(self, X: np.ndarray, y: np.ndarray, num_epochs: int = 1000, lr: float = 0.001):
        """Train model linear regression with gradient descent

        Parameters
        ----------
        X: with shape (num_samples, input_shape)
        y: with shape (num_samples, output_shape)
        num_epochs: number of interactions of gradient descent
        lr: step of linear regression

        Returns
        -------
        None
        """
        m, n = X.shape
        self.init_weights(X.shape[1], y.shape[1])

        for _ in range(num_epochs):
            preds = self.predict(X)
            # compute gradients without loops, only use numpy.
            # IMPORTANT don't forget to compute gradients for L2 regularization
            b_grad = ...  # TODO
            W_grad = ...  # TODO
            self.W = ...  # TODO
            self.b = ...  # TODO

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Do your predictions here :)
        """
        return ...  # TODO


if __name__ == "__main__":
    # TODO run this part of the code to generate plot, save plot and upload to github
    #  you are free to play with parameters
    custom_l2 = CustomLinearRegression(C=10, random_state=42)

    # TODO also use linear regression without L2, implemented in CustomLinearRegression.
    custom_lin_reg = ...
    lg_sk = lg_sklearn()
    ridge = Ridge(alpha=10)

    X, y = make_regression(1000, n_features=1, n_targets=1, random_state=42, noise=0)
    y = np.expand_dims(y, 1)

    # adding anomalous datapoint to see effect of regression
    # you are free to comment this part to see effect on normal linear data
    X = np.vstack((X, np.array([X.max() + 20])))
    y = np.vstack((y, np.array([y.max() + 10])))

    # fitting models
    custom_l2.fit(X, y)
    y_hat_l2 = custom_l2.predict(X)

    custom_lin_reg.fit(X, y)
    y_hat_lin = custom_lin_reg.predict(X)

    lg_sk.fit(X, y)
    y_hat_sk = lg_sk.predict(X)

    ridge.fit(X, y)
    y_hat_ridge = ridge.predict(X)

    # plotting models
    plt.scatter(X, y)
    plt.plot(X, y_hat_l2, color="red", label="Custom L2")
    plt.plot(X, y_hat_lin, color="k", label="Custom Lin reg")
    plt.plot(X, y_hat_sk, color="green", label="Sklearn Lin reg")
    plt.plot(X, y_hat_ridge, color="orange", label="Ridge")
    plt.legend()
    plt.savefig("regressions.png")
