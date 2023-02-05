import numpy as np


def linear_regression(X: np.ndarray, Y: np.ndarray) -> np.ndarray:

    w = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y)
    return w

linear_regression(np.array([[1, 1], [1, 6]]), np.array([1, 2]))