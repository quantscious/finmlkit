import pytest
import numpy as np
import pandas as pd
from finmlkit.bar.base import BarBuilderBase
from finmlkit.bar.data_model import TradesData, FootprintData


class MockBarBuilder(BarBuilderBase):
    """Mock implementation of BarBuilderBase for testing purposes."""

    def __init__(self, trades):
        super().__init__(trades)
        # Set up some predefined close indices for testing
        self._close_indices = np.array([0, 3, 6], dtype=np.int64)
        self._close_ts = np.array([1000000000, 2000000000, 3000000000], dtype=np.int64)

    def _comp_bar_close(self):
        """Simple mock implementation of comp_bar_close."""
        return self._close_ts, self._close_indices


def create_test_trades():
    """Create a simple TradesData object for testing."""
    # Create some simple test data
    timestamps = np.array([1000000000, 1500000000, 1900000000,
                           2100000000, 2500000000, 2900000000,
                           3100000000], dtype=np.int64)

    prices = np.array([100.0, 101.0, 102.0,
                       103.0, 102.0, 101.0,
                       100.0], dtype=np.float64)

    amounts = np.array([1.0, 2.0, 3.0,
                        2.0, 1.0, 2.0,
                        1.0], dtype=np.float64)

    # Alternating buy/sell sides
    sides = np.array([1, -1, 1,
                      -1, 1, -1,
                      1], dtype=np.int8)

    return TradesData(ts=timestamps, px=prices, qty=amounts, side=sides)


def test_build_footprints_returns_correct_type():
    """Test that build_footprints returns a FootprintData object."""
    trades = create_test_trades()
    bar_builder = MockBarBuilder(trades)

    # Call the method
    footprint_data = bar_builder.build_footprints(price_tick_size=0.5)

    # Verify it returns a FootprintData object
    assert isinstance(footprint_data, FootprintData), "Expected a FootprintData object"


def test_build_footprints_includes_new_metrics():
    """Test that build_footprints includes the new metrics in the returned object."""
    trades = create_test_trades()
    bar_builder = MockBarBuilder(trades)

    # Call the method
    footprint_data = bar_builder.build_footprints(price_tick_size=0.5)

    # Verify the new metrics are present
    assert footprint_data.imb_max_run_signed is not None, "imb_max_run_signed should be present"
    assert footprint_data.vp_skew is not None, "vp_skew should be present"
    assert footprint_data.vp_gini is not None, "vp_gini should be present"

    # Check types
    assert isinstance(footprint_data.imb_max_run_signed, np.ndarray), "imb_max_run_signed should be a numpy array"
    assert isinstance(footprint_data.vp_skew, np.ndarray), "vp_skew should be a numpy array"
    assert isinstance(footprint_data.vp_gini, np.ndarray), "vp_gini should be a numpy array"

    # Check dtypes
    assert footprint_data.imb_max_run_signed.dtype == np.int16, "imb_max_run_signed should be int16"
    assert footprint_data.vp_skew.dtype == np.float64, "vp_skew should be float64"
    assert footprint_data.vp_gini.dtype == np.float64, "vp_gini should be float64"

    # Check lengths
    expected_length = len(footprint_data.bar_timestamps)
    assert len(footprint_data.imb_max_run_signed) == expected_length
    assert len(footprint_data.vp_skew) == expected_length
    assert len(footprint_data.vp_gini) == expected_length


def test_footprint_data_slicing_preserves_new_metrics():
    """Test that slicing a FootprintData object preserves the new metrics."""
    trades = create_test_trades()
    bar_builder = MockBarBuilder(trades)

    # Get the footprint data
    footprint_data = bar_builder.build_footprints(price_tick_size=0.5)

    # Slice the data
    sliced_data = footprint_data[0:1]

    # Verify the new metrics are preserved in the sliced data
    assert sliced_data.imb_max_run_signed is not None, "imb_max_run_signed should be preserved after slicing"
    assert sliced_data.vp_skew is not None, "vp_skew should be preserved after slicing"
    assert sliced_data.vp_gini is not None, "vp_gini should be preserved after slicing"

    # Check length of sliced data
    assert len(sliced_data.imb_max_run_signed) == 1, "Sliced imb_max_run_signed should have length 1"
    assert len(sliced_data.vp_skew) == 1, "Sliced vp_skew should have length 1"
    assert len(sliced_data.vp_gini) == 1, "Sliced vp_gini should have length 1"


if __name__ == "__main__":
    pytest.main()