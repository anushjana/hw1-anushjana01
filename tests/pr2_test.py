import numpy as np
from sklearn.datasets import make_regression

from linear_regression import CustomLinearRegression


def test_weights():
    log_reg = CustomLinearRegression(random_state=42, C=0.)
    log_reg.init_weights(10, 10)
    expected_bias = np.zeros((1, 10))
    np.testing.assert_array_equal(log_reg.b, expected_bias)


def test_predict():
    log_reg = CustomLinearRegression(random_state=42, C=0.)
    X, y = make_regression(5, n_features=1, n_targets=1, random_state=42, noise=0)
    y = np.expand_dims(y, 1)
    log_reg.fit(X, y)
    y_hat = log_reg.predict(X)
    expected_y = np.array([2.986544,
                           0.764224,
                           3.514930,
                           6.578481,
                           0.428628])
    expected_y = np.expand_dims(expected_y, 1)
    np.testing.assert_allclose(y_hat, expected_y, rtol=1e-6)
