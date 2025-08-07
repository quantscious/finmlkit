import numpy as np
import pytest
import os
# disable jit
#os.environ['NUMBA_DISABLE_JIT'] = "1"

from finmlkit.label.weights import class_balance_weights


def test_class_balance_weights_basic():
    """Test basic functionality with uniform weights."""
    labels = np.array([1, -1, 1, 0, -1, 1], dtype=np.int8)
    weights = np.ones_like(labels, dtype=np.float64)

    unique_labels, class_weights, sum_w_class, final_weights = class_balance_weights(labels, weights)

    # Expected results
    expected_unique_labels = np.array([-1, 0, 1], dtype=np.int8)  # np.unique returns sorted
    expected_sum_w_class = np.array([2.0, 1.0, 3.0], dtype=np.float64)  # [-1: 2 samples, 0: 1 sample, 1: 3 samples]
    total_weights = 6.0  # CORRECTED: sum of all weights
    expected_class_weights = np.array([
        total_weights / (3 * 2.0),  # 6 / (3 * 2) = 1.0
        total_weights / (3 * 1.0),  # 6 / (3 * 1) = 2.0
        total_weights / (3 * 3.0)  # 6 / (3 * 3) = 0.6667
    ], dtype=np.float64)

    # FIXED: Don't sort - maintain order from np.unique
    np.testing.assert_array_equal(unique_labels, expected_unique_labels)
    np.testing.assert_array_almost_equal(sum_w_class, expected_sum_w_class)
    np.testing.assert_array_almost_equal(class_weights, expected_class_weights)

    # Verify final weights calculation
    expected_final_weights = weights * np.array([
        expected_class_weights[2],  # label 1 -> class_weights[2]
        expected_class_weights[0],  # label -1 -> class_weights[0]
        expected_class_weights[2],  # label 1 -> class_weights[2]
        expected_class_weights[1],  # label 0 -> class_weights[1]
        expected_class_weights[0],  # label -1 -> class_weights[0]
        expected_class_weights[2]  # label 1 -> class_weights[2]
    ])
    np.testing.assert_array_almost_equal(final_weights, expected_final_weights)


def test_class_balance_weights_with_custom_weights():
    """Test with non-uniform base weights."""
    labels = np.array([1, -1, 1, 0, -1, 1], dtype=np.int8)
    weights = np.array([1.0, 2.0, 1.0, 3.0, 2.0, 1.0], dtype=np.float64)

    unique_labels, class_weights, sum_w_class, final_weights = class_balance_weights(labels, weights)

    # Calculate expected values
    # Label -1: indices 1,4 -> weights 2.0 + 2.0 = 4.0
    # Label 0:  index 3   -> weight 3.0
    # Label 1:  indices 0,2,5 -> weights 1.0 + 1.0 + 1.0 = 3.0
    expected_sum_w_class = np.array([4.0, 3.0, 3.0], dtype=np.float64)
    total_weights = 10.0  # CORRECTED: actual sum
    expected_class_weights = np.array([
        total_weights / (3 * 4.0),  # 10 / (3 * 4) = 0.8333
        total_weights / (3 * 3.0),  # 10 / (3 * 3) = 1.1111
        total_weights / (3 * 3.0)  # 10 / (3 * 3) = 1.1111
    ], dtype=np.float64)

    np.testing.assert_array_equal(unique_labels, np.array([-1, 0, 1], dtype=np.int8))
    np.testing.assert_array_almost_equal(sum_w_class, expected_sum_w_class)
    np.testing.assert_array_almost_equal(class_weights, expected_class_weights)


def test_class_balance_weights_zero_weights_no_error():
    """Test that zero weights are handled gracefully without errors."""
    labels = np.array([1, -1, 1, 0, -1, 1], dtype=np.int8)
    weights = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)

    # CORRECTED: Function handles zeros gracefully, doesn't raise errors
    unique_labels, class_weights, sum_w_class, final_weights = class_balance_weights(labels, weights)

    # All class weights should be 0 when sum_w_class[c] = 0
    expected_class_weights = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    expected_sum_w_class = np.array([0.0, 0.0, 0.0], dtype=np.float64)

    np.testing.assert_array_equal(unique_labels, np.array([-1, 0, 1], dtype=np.int8))
    np.testing.assert_array_almost_equal(class_weights, expected_class_weights)
    np.testing.assert_array_almost_equal(sum_w_class, expected_sum_w_class)
    np.testing.assert_array_almost_equal(final_weights, np.zeros_like(weights))


def test_class_balance_weights_single_class():
    """Test with single class."""
    labels = np.array([1, 1, 1], dtype=np.int8)
    weights = np.array([1.0, 1.0, 1.0], dtype=np.float64)

    unique_labels, class_weights, sum_w_class, final_weights = class_balance_weights(labels, weights)

    expected_unique_labels = np.array([1], dtype=np.int8)
    expected_sum_w_class = np.array([3.0], dtype=np.float64)
    expected_class_weights = np.array([3.0 / (1 * 3.0)], dtype=np.float64)  # = [1.0]

    np.testing.assert_array_equal(unique_labels, expected_unique_labels)
    np.testing.assert_array_almost_equal(sum_w_class, expected_sum_w_class)
    np.testing.assert_array_almost_equal(class_weights, expected_class_weights)
    np.testing.assert_array_almost_equal(final_weights, weights)  # Should equal input weights


def test_class_balance_weights_partial_zero_class():
    """Test with some classes having zero weights."""
    labels = np.array([1, -1, 1, 0, -1, 1], dtype=np.int8)
    weights = np.array([1.0, 0.0, 1.0, 0.0, 0.0, 1.0], dtype=np.float64)  # Only label 1 has weight

    unique_labels, class_weights, sum_w_class, final_weights = class_balance_weights(labels, weights)

    expected_sum_w_class = np.array([0.0, 0.0, 3.0], dtype=np.float64)  # Only class 1 has weight
    total_weights = 3.0
    expected_class_weights = np.array([
        0.0,  # Class -1 has zero weight
        0.0,  # Class 0 has zero weight
        total_weights / (3 * 3.0)  # Class 1: 3 / (3 * 3) = 0.3333
    ], dtype=np.float64)

    np.testing.assert_array_almost_equal(class_weights, expected_class_weights)
    np.testing.assert_array_almost_equal(sum_w_class, expected_sum_w_class)


def test_class_balance_weights_single_sample():
    """Test with single sample."""
    labels = np.array([5], dtype=np.int8)
    weights = np.array([2.5], dtype=np.float64)

    unique_labels, class_weights, sum_w_class, final_weights = class_balance_weights(labels, weights)

    np.testing.assert_array_equal(unique_labels, np.array([5], dtype=np.int8))
    np.testing.assert_array_almost_equal(sum_w_class, np.array([2.5], dtype=np.float64))
    np.testing.assert_array_almost_equal(class_weights, np.array([1.0], dtype=np.float64))  # 2.5 / (1 * 2.5)
    np.testing.assert_array_almost_equal(final_weights, np.array([2.5], dtype=np.float64))


def test_class_balance_weights_negative_labels():
    """Test with various negative labels."""
    labels = np.array([-5, -10, -5, -2, -10], dtype=np.int8)
    weights = np.array([1.0, 2.0, 1.0, 3.0, 2.0], dtype=np.float64)

    unique_labels, class_weights, sum_w_class, final_weights = class_balance_weights(labels, weights)

    expected_unique_labels = np.array([-10, -5, -2], dtype=np.int8)  # Sorted by np.unique
    expected_sum_w_class = np.array([4.0, 2.0, 3.0], dtype=np.float64)  # [-10: 4.0, -5: 2.0, -2: 3.0]
    total_weights = 9.0

    np.testing.assert_array_equal(unique_labels, expected_unique_labels)
    np.testing.assert_array_almost_equal(sum_w_class, expected_sum_w_class)

    # Verify final weights maintain relationships
    assert len(final_weights) == len(labels)
    assert np.sum(final_weights) > 0


def test_class_balance_weights_imbalanced_scenario():
    """Test realistic imbalanced dataset scenario."""
    # Simulate imbalanced dataset: many 0s, few 1s, very few -1s
    labels = np.array([0, 0, 0, 0, 0, 0, 1, 1, -1], dtype=np.int8)
    weights = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 2.0, 2.0, 5.0], dtype=np.float64)

    unique_labels, class_weights, sum_w_class, final_weights = class_balance_weights(labels, weights)

    expected_sum_w_class = np.array([5.0, 3.0, 4.0], dtype=np.float64)  # [-1: 5.0, 0: 3.0, 1: 4.0]
    total_weights = 12.0
    expected_class_weights = np.array([
        total_weights / (3 * 5.0),  # -1: 12/(3*5) = 0.8
        total_weights / (3 * 3.0),  # 0:  12/(3*3) = 1.3333
        total_weights / (3 * 4.0)  # 1:  12/(3*4) = 1.0
    ], dtype=np.float64)

    np.testing.assert_array_almost_equal(class_weights, expected_class_weights, decimal=4)

    # Verify class balancing effect: minority class (-1) gets lower weight due to higher base weight
    # but the balancing formula adjusts for this
    assert class_weights[1] > class_weights[0]  # Class 0 should get higher weight (most samples)


def test_class_balance_weights_mathematical_properties():
    """Test mathematical properties of class balancing."""
    labels = np.array([1, 2, 1, 2, 1], dtype=np.int8)
    weights = np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float64)

    unique_labels, class_weights, sum_w_class, final_weights = class_balance_weights(labels, weights)

    # Test that class weights inversely relate to class frequency
    # Class 1: 3 samples, Class 2: 2 samples
    # So class 2 should have higher weight than class 1
    class_1_idx = np.where(unique_labels == 1)[0][0]
    class_2_idx = np.where(unique_labels == 2)[0][0]
    assert class_weights[class_2_idx] > class_weights[class_1_idx]

    # Test that sum of weighted final weights per class is balanced
    final_weight_sum_1 = np.sum(final_weights[labels == 1])
    final_weight_sum_2 = np.sum(final_weights[labels == 2])

    # After class balancing, weighted sums should be equal (balanced)
    np.testing.assert_almost_equal(final_weight_sum_1, final_weight_sum_2, decimal=10)


def test_class_balance_weights_data_types():
    """Test correct data types in outputs."""
    labels = np.array([0, 1], dtype=np.int8)
    weights = np.array([1.0, 2.0], dtype=np.float64)

    unique_labels, class_weights, sum_w_class, final_weights = class_balance_weights(labels, weights)

    # Verify output data types
    assert unique_labels.dtype == np.int8
    assert class_weights.dtype == np.float64
    assert sum_w_class.dtype == np.float64
    assert final_weights.dtype == np.float64

    # Verify array shapes
    assert len(unique_labels) == 2
    assert len(class_weights) == 2
    assert len(sum_w_class) == 2
    assert len(final_weights) == 2


def test_class_balance_weights_large_dataset():
    """Test with larger dataset for performance validation."""
    np.random.seed(42)
    n_samples = 10000

    # Create imbalanced dataset
    labels = np.random.choice([-1, 0, 1], n_samples, p=[0.1, 0.8, 0.1]).astype(np.int8)
    weights = np.random.exponential(1.0, n_samples)

    unique_labels, class_weights, sum_w_class, final_weights = class_balance_weights(labels, weights)

    # Should complete without error
    assert len(unique_labels) == 3
    assert len(class_weights) == 3
    assert len(sum_w_class) == 3
    assert len(final_weights) == n_samples

    # All outputs should be finite
    assert np.all(np.isfinite(class_weights))
    assert np.all(np.isfinite(sum_w_class))
    assert np.all(np.isfinite(final_weights))

    # Class with fewer samples should generally get higher weight
    minority_classes = np.where(sum_w_class < np.mean(sum_w_class))[0]
    majority_classes = np.where(sum_w_class > np.mean(sum_w_class))[0]

    if len(minority_classes) > 0 and len(majority_classes) > 0:
        assert np.mean(class_weights[minority_classes]) > np.mean(class_weights[majority_classes])


def test_class_balance_weights_edge_values():
    """Test with edge values and extreme weights."""
    labels = np.array([0, 1, 0, 1], dtype=np.int8)
    weights = np.array([1e-10, 1e10, 1e-10, 1e10], dtype=np.float64)  # Extreme weights

    unique_labels, class_weights, sum_w_class, final_weights = class_balance_weights(labels, weights)

    # Should handle extreme values without overflow/underflow
    assert np.all(np.isfinite(class_weights))
    assert np.all(np.isfinite(final_weights))
    assert np.all(class_weights >= 0)


if __name__ == "__main__":
    pytest.main([__file__])