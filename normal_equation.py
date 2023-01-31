import numpy as np


def linear_regression(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    Fit linear regression to the data of form Y = w * X
    **Note** that there is not separate term for the intercept.

    Example
    >>> linear_regression(np.array([[1, 1], [1, 6]]), np.array([1, 2]))
    array([0.8, 0.2])
    """
    w = ... # TODO
    return w
