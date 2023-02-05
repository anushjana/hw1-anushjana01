import numpy as np
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression as lg_sklearn
from sklearn.linear_model import Ridge

from typing import Union

class CustomLinearRegression:
    def __init__(self, C: Union[float, int] = 0, random_state: int = 42):
        self.random_state = random_state
        self.C = C
        self.W = None
        self.b = None

    def init_weights(self, input_size: int, output_size: int):

        np.random.seed(self.random_state)
        self.W = np.random.normal(0, 0.01, (input_size, output_size))
        self.b = np.zeros((1, output_size))

    def fit(self, X: np.ndarray, y: np.ndarray, num_epochs: int = 1000, lr: float = 0.001):

        m, n = X.shape
        self.init_weights(X.shape[1], y.shape[1])

        for _ in range(num_epochs):
            preds = self.predict(X)
            b_grad = 2 / m * np.sum(preds - y) + 2 * self.C * self.b
            W_grad = 2 / m * X.T.dot(preds - y) + 2 * self.C * self.W
            self.W = self.W - lr * W_grad
            self.b = self.b - lr * b_grad
    def predict(self, X: np.ndarray) -> np.ndarray:

        return X.dot(self.W) + self.b


if __name__ == "__main__":

    custom_l2 = CustomLinearRegression(C=10, random_state=42)
    custom_lin_reg = CustomLinearRegression(random_state=42)
    lg_sk = lg_sklearn()
    ridge = Ridge(alpha=10)

    X, y = make_regression(1000, n_features=1, n_targets=1, random_state=42, noise=0)
    y = np.expand_dims(y, 1)

    X = np.vstack((X, np.array([X.max() + 20])))
    y = np.vstack((y, np.array([y.max() + 10])))


    custom_l2.fit(X, y)
    y_hat_l2 = custom_l2.predict(X)

    custom_lin_reg.fit(X, y)
    y_hat_lin = custom_lin_reg.predict(X)

    lg_sk.fit(X, y)
    y_hat_sk = lg_sk.predict(X)

    ridge.fit(X, y)
    y_hat_ridge = ridge.predict(X)

    plt.scatter(X, y)
    plt.plot(X, y_hat_l2, color="red", label="Custom L2")
    plt.plot(X, y_hat_lin, color="k", label="Custom Lin reg")
    plt.plot(X, y_hat_sk, color="green", label="Sklearn Lin reg")
    plt.plot(X, y_hat_ridge, color="orange", label="Ridge")
    plt.legend()
    plt.savefig("regressions.png")
