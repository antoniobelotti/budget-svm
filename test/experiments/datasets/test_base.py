import unittest

import numpy as np

from experiments.datasets.Base import compute_gram_matrix
from kernel import GaussianKernel


class TestBase(unittest.TestCase):
    def test_compute_gram_matrix(self):
        kernel_obj = GaussianKernel()
        k = kernel_obj.compute

        # "training set"
        x1 = np.random.rand(1, 2)
        x2 = np.random.rand(1, 2)
        x3 = np.random.rand(1, 2)
        X = np.array([x1, x2, x3])

        expected = np.array(
            [
                [k(x1, x1), k(x1, x2), k(x1, x3)],
                [k(x2, x1), k(x2, x2), k(x2, x3)],
                [k(x3, x1), k(x3, x2), k(x3, x3)],
            ]
        )

        gram = compute_gram_matrix(X, X, kernel_obj)

        np.testing.assert_array_equal(expected, gram)

        # "test set"
        xt1 = np.random.rand(1, 2)
        xt2 = np.random.rand(1, 2)
        X_test = np.array([xt1, xt2])

        # shape should be (n_sample_test, n_samples_train)
        expected_test = np.array(
            [
                [k(xt1, x1), k(xt1, x2), k(xt1, x3)],
                [k(xt2, x1), k(xt2, x2), k(xt2, x3)],
            ]
        )

        gram_test = compute_gram_matrix(X, X_test, kernel_obj)
        np.testing.assert_array_equal(expected_test, gram_test)
