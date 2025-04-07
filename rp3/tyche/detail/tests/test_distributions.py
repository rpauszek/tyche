import numpy as np
from rp3.tyche.detail.distributions import Normal


def test_normal():
    n = Normal([0, 1, 2], [20, 15, 10])

    np.testing.assert_allclose(n.mu, [0, 1, 2])
    np.testing.assert_allclose(n.tau, [20, 15, 10])
    np.testing.assert_allclose(n.var, 1 / np.array([20, 15, 10]))
    np.testing.assert_allclose(n.sigma, np.sqrt(1 / np.array([20, 15, 10])))
    assert n.K == 3

    assert isinstance(n.mu, np.ndarray)
    assert isinstance(n.tau, np.ndarray)

    x = [-0.5, 1.5, 2.5]
    expected_pdf = [
        [1.46449826e-01, 3.01855589e-10, 1.28238947e-27],
        [7.24830256e-08, 2.36948270e-01, 7.24830256e-08],
        [3.38226403e-14, 3.61444785e-01, 3.61444785e-01],
    ]
    for row, expected in zip(n.pdf(x), expected_pdf):
        np.testing.assert_allclose(row, expected)
