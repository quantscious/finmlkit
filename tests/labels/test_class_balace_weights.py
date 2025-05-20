import numpy as np
import pytest
import os
# disable jit
#os.environ['NUMBA_DISABLE_JIT'] = "1"

from finmlkit.label.weights import class_balance_weights


def test_class_balance_weights_basic():
    labels = np.array([1, -1, 1, 0, -1, 1], dtype=np.int8)
    weights = np.ones_like(labels, dtype=np.float64)

    unique_labels, class_weights, n_class_elements, _ = class_balance_weights(labels, weights)

    expected_unique_labels = np.array([-1, 0, 1], dtype=np.int8)
    expected_n_class_elements = np.array([2.0, 1.0, 3.0], dtype=np.float64)
    expected_class_weights = np.array([6 / (3 * 2), 6 / (3 * 1), 6 / (3 * 3)], dtype=np.float64)

    np.testing.assert_array_equal(np.sort(unique_labels), np.sort(expected_unique_labels))
    np.testing.assert_almost_equal(np.sort(n_class_elements), np.sort(expected_n_class_elements))
    np.testing.assert_almost_equal(np.sort(class_weights), np.sort(expected_class_weights))


def test_class_balance_weights_with_weights():
    labels = np.array([1, -1, 1, 0, -1, 1], dtype=np.int8)
    weights = np.array([1.0, 2.0, 1.0, 3.0, 2.0, 1.0], dtype=np.float64)

    unique_labels, class_weights, n_class_elements, _ = class_balance_weights(labels, weights)

    expected_unique_labels = np.array([-1, 0, 1], dtype=np.int8)
    expected_n_class_elements = np.array([4.0, 3.0, 3.0], dtype=np.float64)
    expected_class_weights = np.array([6 / (3 * 4), 6 / (3 * 3), 6 / (3 * 3)], dtype=np.float64)

    np.testing.assert_array_equal(np.sort(unique_labels), np.sort(expected_unique_labels))
    np.testing.assert_almost_equal(np.sort(n_class_elements), np.sort(expected_n_class_elements))
    np.testing.assert_almost_equal(np.sort(class_weights), np.sort(expected_class_weights))


def test_class_balance_weights_zero_weights():
    labels = np.array([1, -1, 1, 0, -1, 1], dtype=np.int8)
    weights = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)

    with pytest.raises(ZeroDivisionError):
        class_balance_weights(labels, weights)


def test_class_balance_weights_single_class():
    labels = np.array([1, 1, 1], dtype=np.int8)
    weights = np.array([1.0, 1.0, 1.0], dtype=np.float64)

    unique_labels, class_weights, n_class_elements, _ = class_balance_weights(labels, weights)

    expected_unique_labels = np.array([1], dtype=np.int8)
    expected_n_class_elements = np.array([3.0], dtype=np.float64)
    expected_class_weights = np.array([3 / (1 * 3)], dtype=np.float64)

    np.testing.assert_array_equal(unique_labels, expected_unique_labels)
    np.testing.assert_almost_equal(n_class_elements, expected_n_class_elements)
    np.testing.assert_almost_equal(class_weights, expected_class_weights)