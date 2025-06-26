import pytest
import pandas as pd
import numpy as np
from finmlkit.feature.kit import Feature
from finmlkit.feature.transforms import Identity


def generate_test_df():
    """Generate a test DataFrame with simple OHLC data"""
    dates = pd.date_range("2025-01-01", periods=100, freq="1h")
    data = {
        "open": np.sin(np.linspace(0, 10, 100)) + 5,
        "high": np.sin(np.linspace(0, 10, 100)) + 6,
        "low": np.sin(np.linspace(0, 10, 100)) + 4,
        "close": np.sin(np.linspace(0.2, 10.2, 100)) + 5.1
    }
    return pd.DataFrame(data, index=dates)


class TestMinMaxOps:
    """Test min and max operations between features"""

    def setup_method(self):
        """Set up test fixtures before each test method"""
        self.df = generate_test_df()
        self.open = Feature(Identity("open"))
        self.high = Feature(Identity("high"))
        self.low = Feature(Identity("low"))
        self.close = Feature(Identity("close"))

    def test_min_feature_feature(self):
        """Test min operation between two features"""
        min_feature = Feature.min(self.open, self.close)
        result = min_feature(self.df)

        # Calculate the expected result
        expected = np.minimum(self.df["open"], self.df["close"])

        # Assert that results match
        np.testing.assert_array_almost_equal(result.values, expected.values)

    def test_max_feature_feature(self):
        """Test max operation between two features"""
        max_feature = Feature.max(self.open, self.close)
        result = max_feature(self.df)

        # Calculate the expected result
        expected = np.maximum(self.df["open"], self.df["close"])

        # Assert that results match
        np.testing.assert_array_almost_equal(result.values, expected.values)

    def test_min_feature_constant(self):
        """Test min operation between a feature and a constant"""
        constant = 5.0
        min_feature = Feature.min(self.high, constant)
        result = min_feature(self.df)

        # Calculate the expected result
        expected = np.minimum(self.df["high"], constant)

        # Assert that results match
        np.testing.assert_array_almost_equal(result.values, expected.values)

    def test_max_feature_constant(self):
        """Test max operation between a feature and a constant"""
        constant = 5.0
        max_feature = Feature.max(self.low, constant)
        result = max_feature(self.df)

        # Calculate the expected result
        expected = np.maximum(self.df["low"], constant)

        # Assert that results match
        np.testing.assert_array_almost_equal(result.values, expected.values)

    def test_min_constant_feature(self):
        """Test min operation between a constant and a feature"""
        constant = 5.0
        min_feature = Feature.min(constant, self.high)
        result = min_feature(self.df)

        # Calculate the expected result
        expected = np.minimum(constant, self.df["high"])

        # Assert that results match
        np.testing.assert_array_almost_equal(result.values, expected.values)

    def test_max_constant_feature(self):
        """Test max operation between a constant and a feature"""
        constant = 5.0
        max_feature = Feature.max(constant, self.low)
        result = max_feature(self.df)

        # Calculate the expected result
        expected = np.maximum(constant, self.df["low"])

        # Assert that results match
        np.testing.assert_array_almost_equal(result.values, expected.values)

    def test_complex_expression(self):
        """Test a complex expression using min/max operations like the wick_up_ratio example"""
        # wick_up_ratio = (high - max(open, close)) / (high - low + 1e-12)
        max_open_close = Feature.max(self.open, self.close)
        high_minus_max = self.high - max_open_close
        denominator = self.high - self.low + 1e-12
        wick_up_ratio = high_minus_max / denominator

        result = wick_up_ratio(self.df)

        # Calculate the expected result directly
        max_oc = np.maximum(self.df["open"], self.df["close"])
        expected = (self.df["high"] - max_oc) / (self.df["high"] - self.df["low"] + 1e-12)

        # Assert that results match
        np.testing.assert_array_almost_equal(result.values, expected.values)

    def test_chained_min_max(self):
        """Test chaining min and max operations"""
        # min(max(open, close), high)
        max_open_close = Feature.max(self.open, self.close)
        min_max_high = Feature.min(max_open_close, self.high)

        result = min_max_high(self.df)

        # Calculate the expected result
        max_oc = np.maximum(self.df["open"], self.df["close"])
        expected = np.minimum(max_oc, self.df["high"])

        # Assert that results match
        np.testing.assert_array_almost_equal(result.values, expected.values)
