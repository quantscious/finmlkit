import pytest
import pandas as pd
import numpy as np
from finmlkit.feature.transforms import DirRunLen
import datetime as dt

class TestDirRunLen:
    def setup_method(self):
        # Create synthetic test data with different return patterns
        index = pd.date_range("2023-01-01", periods=20, freq="1h")

        # Case 1: Simple alternating signs
        alternating_returns = [0.01, 0.02, 0.03, -0.01, -0.02, 0.04, 0.05, 0, 0.02, -0.03]
        alternating_returns = alternating_returns + alternating_returns  # Repeat to get 20 values

        # Create dataframe
        self.df = pd.DataFrame({
            'ret1': alternating_returns[:20]
        }, index=index)

        # Create transform
        self.transform = DirRunLen(input_col="ret1")

    def test_basic_functionality(self):
        """Test that DirRunLen counts consecutive same-sign returns correctly."""
        result = self.transform._pd(self.df)

        # Expected run lengths for our data:
        # [0, 1, 2, 3, 1, 2, 1, 2, 1, 1, 1, 2, 3, 1, 2, 1, 2, 1, 1, 1]
        # First value is 0 (no streak yet)
        expected = pd.Series([0, 1, 2, 3, 1, 2, 1, 2, 0, 1, 1, 2, 3, 1, 2, 1, 2, 0, 1, 1],
                             index=self.df.index,
                             name='ret1_dir_run_len', dtype=np.int8)

        pd.testing.assert_series_equal(result, expected)

    def test_all_positive_returns(self):
        """Test with all positive returns."""
        df_positive = pd.DataFrame({
            'ret1': [0.01, 0.02, 0.03, 0.04, 0.05]
        }, index=pd.date_range("2023-01-01", periods=5, freq="1h"))

        result = self.transform._pd(df_positive)

        # Expected: 0, 1, 2, 3, 4
        expected = pd.Series([0, 1, 2, 3, 4], index=df_positive.index, name='ret1_dir_run_len')
        pd.testing.assert_series_equal(result, expected)

    def test_all_negative_returns(self):
        """Test with all negative returns."""
        df_negative = pd.DataFrame({
            'ret1': [-0.01, -0.02, -0.03, -0.04, -0.05]
        }, index=pd.date_range("2023-01-01", periods=5, freq="1h"))

        result = self.transform._pd(df_negative)

        # Expected: 0, 1, 2, 3, 4
        expected = pd.Series([0, 1, 2, 3, 4], index=df_negative.index, name='ret1_dir_run_len')
        pd.testing.assert_series_equal(result, expected)

    def test_with_zeros(self):
        """Test handling of zero returns."""
        df_zeros = pd.DataFrame({
            'ret1': [0.01, 0.02, 0, 0, 0.03, 0.04, 0, -0.01]
        }, index=pd.date_range("2023-01-01", periods=8, freq="1h"))

        result = self.transform._pd(df_zeros)

        # Expected: 0, 1, 2, 0, 0, 1, 2, 0
        expected = pd.Series([0, 1, 2, 0, 0, 1, 2, 0], index=df_zeros.index, name='ret1_dir_run_len')
        pd.testing.assert_series_equal(result, expected)

    def test_random_data(self):
        """Test with random data."""
        np.random.seed(42)
        random_returns = np.random.normal(0, 0.01, 50)

        df_random = pd.DataFrame({
            'ret1': random_returns
        }, index=pd.date_range("2023-01-01", periods=50, freq="1h"))

        result = self.transform._pd(df_random)

        # Check basic properties
        assert len(result) == len(df_random)
        assert result.iloc[0] == 0  # First value should be 0

        # Verify that the counts are correct
        for i in range(1, len(result)):
            # If sign changes or return is 0, count should reset
            if (np.sign(df_random['ret1'].iloc[i]) != np.sign(df_random['ret1'].iloc[i-1]) or
                df_random['ret1'].iloc[i] == 0):
                assert result.iloc[i] == 1 or result.iloc[i] == 0
            # Otherwise, count should increase
            else:
                assert result.iloc[i] == result.iloc[i-1] + 1

    def test_edge_cases(self):
        """Test edge cases like NaN and very small values."""
        df_edge = pd.DataFrame({
            'ret1': [0.01, np.nan, 0.03, 1e-10, -1e-10, -0.02]
        }, index=pd.date_range("2023-01-01", periods=6, freq="1h"))

        result = self.transform._pd(df_edge)

        # Expected behavior with NaN: treated as a break in the sequence
        expected = pd.Series([0, 1, 0, 1, 0, 1], index=df_edge.index, name='ret1_dir_run_len')
        pd.testing.assert_series_equal(result, expected)


if __name__ == "__main__":
    pytest.main([__file__])
