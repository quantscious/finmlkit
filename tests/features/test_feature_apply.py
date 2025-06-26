import pytest
import pandas as pd
import numpy as np
from finmlkit.feature.kit import Feature
from finmlkit.feature.base import SISOTransform
from finmlkit.feature.transforms import Identity

class TestFeatureApply:
    """Test the apply method and other new methods added to Feature class."""

    @pytest.fixture
    def simple_df(self):
        """Create a simple DataFrame for testing."""
        return pd.DataFrame({
            'close': [100, 110, 105, 120, 115, 125, 130, 120, 110, 115],
            'volume': [1000, 1200, 900, 1500, 1100, 1300, 1400, 1000, 800, 1200]
        }, index=pd.date_range(start='2025-01-01', periods=10, freq='D'))

    @pytest.fixture
    def feature(self, simple_df):
        """Create a Feature instance for testing."""
        transform = Identity("close")
        return Feature(transform)

    def test_apply_basic(self, feature, simple_df):
        """Test basic apply functionality."""
        # Define a simple function to apply
        def double(x):
            return x * 2

        # Apply the function
        result_feature = feature.apply(double)

        # Check the result
        result = result_feature(simple_df)
        expected = simple_df['close'] * 2

        np.testing.assert_array_equal(result.values, expected.values)
        assert result.name == "close_double"

    def test_apply_with_args(self, feature, simple_df):
        """Test apply with additional arguments."""
        # Apply function with args
        result_feature = feature.apply(lambda x, factor: x * factor, 3)

        # Check the result
        result = result_feature(simple_df)
        expected = simple_df['close'] * 3

        np.testing.assert_array_equal(result.values, expected.values)

    def test_apply_with_kwargs(self, feature, simple_df):
        """Test apply with keyword arguments."""
        # Apply function with kwargs
        result_feature = feature.apply(lambda x, factor=1: x * factor, factor=4)

        # Check the result
        result = result_feature(simple_df)
        expected = simple_df['close'] * 4

        np.testing.assert_array_equal(result.values, expected.values)

    def test_apply_with_custom_suffix(self, feature, simple_df):
        """Test apply with custom suffix."""
        # Apply function with custom suffix
        result_feature = feature.apply(lambda x: x * 2, suffix="doubled")

        # Check the result
        result = result_feature(simple_df)

        assert result.name == "close_doubled"

    def test_clip(self, feature, simple_df):
        """Test clip method."""
        # Test clip with lower bound
        result_feature = feature.clip(lower=110)
        result = result_feature(simple_df)
        expected = simple_df['close'].clip(lower=110)

        np.testing.assert_array_equal(result.values, expected.values)

        # Test clip with upper bound
        result_feature = feature.clip(upper=115)
        result = result_feature(simple_df)
        expected = simple_df['close'].clip(upper=115)

        np.testing.assert_array_equal(result.values, expected.values)

        # Test clip with both bounds
        result_feature = feature.clip(lower=105, upper=120)
        result = result_feature(simple_df)
        expected = simple_df['close'].clip(lower=105, upper=120)

        np.testing.assert_array_equal(result.values, expected.values)

    def test_abs(self, feature, simple_df):
        """Test abs method."""
        # Create a feature with negative values
        df_with_neg = simple_df.copy()
        df_with_neg['close'] = df_with_neg['close'] - 115  # Some values will be negative

        result_feature = feature.abs()
        result = result_feature(df_with_neg)
        expected = df_with_neg['close'].abs()

        np.testing.assert_array_equal(result.values, expected.values)

    def test_log(self, feature, simple_df):
        """Test log method."""
        result_feature = feature.log()
        result = result_feature(simple_df)
        expected = simple_df['close'].apply(lambda v: np.log(v) if v > 0 else np.nan)

        np.testing.assert_array_almost_equal(result.values, expected.values)

    def test_exp(self, feature, simple_df):
        """Test exp method."""
        result_feature = feature.exp()
        result = result_feature(simple_df)
        expected = simple_df['close'].apply(np.exp)

        np.testing.assert_array_almost_equal(result.values, expected.values)

    def test_square(self, feature, simple_df):
        """Test square method."""
        result_feature = feature.square()
        result = result_feature(simple_df)
        expected = simple_df['close'] ** 2

        np.testing.assert_array_equal(result.values, expected.values)

    def test_sqrt(self, feature, simple_df):
        """Test sqrt method."""
        result_feature = feature.sqrt()
        result = result_feature(simple_df)
        expected = simple_df['close'].apply(lambda v: np.sqrt(v) if v >= 0 else np.nan)

        np.testing.assert_array_almost_equal(result.values, expected.values)

    def test_rolling_mean(self, feature, simple_df):
        """Test rolling_mean method."""
        window = 3
        result_feature = feature.rolling_mean(window)
        result = result_feature(simple_df)
        expected = simple_df['close'].rolling(window=window).mean()

        # Check only non-NaN values (first window-1 values will be NaN)
        np.testing.assert_array_almost_equal(
            result.values[window-1:],
            expected.values[window-1:]
        )

    def test_rolling_std(self, feature, simple_df):
        """Test rolling_std method."""
        window = 3
        result_feature = feature.rolling_std(window)
        result = result_feature(simple_df)
        expected = simple_df['close'].rolling(window=window).std()

        # Check only non-NaN values (first window-1 values will be NaN)
        np.testing.assert_array_almost_equal(
            result.values[window-1:],
            expected.values[window-1:]
        )

    def test_chained_operations(self, feature, simple_df):
        """Test chaining multiple operations."""
        # Create a more complex feature with chained operations
        result_feature = (feature - 100).abs().clip(lower=5).rolling_mean(2)

        # Calculate expected result step by step
        step1 = simple_df['close'] - 100  # Subtract 100
        step2 = step1.abs()              # Apply abs
        step3 = step2.clip(lower=5)      # Clip with lower=5
        expected = step3.rolling(window=2).mean()  # Rolling mean with window=2

        # Get actual result
        result = result_feature(simple_df)

        # Compare non-NaN values
        np.testing.assert_array_almost_equal(
            result.values[1:],  # First value will be NaN due to rolling
            expected.values[1:]
        )

    def test_complex_example(self, simple_df):
        """Test a complex example similar to the one in the original request."""
        # Create two features
        rv_12 = Feature(Identity("close"))
        bv_12 = Feature(Identity("volume"))

        # Apply operation: (rv_12 - bv_12).clip(lower=0)
        # Note: This doesn't make sense in practice (comparing price and volume)
        # but it's just for testing the API
        scaled_bv_12 = bv_12.apply(lambda x: x / 10)  # Scale down volume to be comparable with price
        result_feature = (rv_12 - scaled_bv_12).clip(lower=0)

        # Calculate expected result
        diff = simple_df['close'] - (simple_df['volume'] / 10)
        expected = diff.clip(lower=0)

        # Get actual result
        result = result_feature(simple_df)

        # Compare
        np.testing.assert_array_almost_equal(result.values, expected.values)
