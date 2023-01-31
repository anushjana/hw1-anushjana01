import pytest
import numpy as np
from normal_equation import linear_regression


@pytest.mark.parametrize("X, y, expected_w", [
    (np.array([[1, 1, 1, 1], [2104, 1416, 1534, 852], [5, 3, 3, 2], [1, 2, 2, 1], [45, 41, 30, 36]]).T,
     np.array([460, 232, 315, 178]), np.array([-7.21308594e+01, 5.35469055e-02, 2.16058594e+02, -6.91113281e+00,
                                               -7.29681396e+00])),

    (np.array([[1, 1], [1, 6]]), np.array([1, 2]), np.array([0.8, 0.2])),

])
def test_linear_regression_normal_equation(X, y, expected_w):
    assert np.allclose(linear_regression(X, y), expected_w, rtol=1e-2, atol=1e-2)
