"""
Tests for time-based features in the finmlkit.feature.core.time module
"""
import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timezone, timedelta
from numpy.testing import assert_array_almost_equal, assert_array_equal

from finmlkit.feature.core.time import time_cues


def create_timestamp_array(start_date, periods, freq='D'):
    """Helper function to create timestamp arrays for testing"""
    dates = pd.date_range(start=start_date, periods=periods, freq=freq)
    return np.array([int(dt.timestamp() * 1_000_000_000) for dt in dates.to_pydatetime()])


def test_time_cues_day_of_week():
    """Test that day of week features are correctly calculated"""
    # Create a week of timestamps starting from Monday, Jan 6, 2020 (a Monday)
    # This ensures we cover each day of the week
    start_date = datetime(2020, 1, 6, tzinfo=timezone.utc)
    timestamps = create_timestamp_array(start_date, periods=7)

    # Run the time_cues function
    sin_td, cos_td, sin_dw, cos_dw, asia, eu, us, trans, top_hr = time_cues(timestamps)

    # Expected days of week for the timestamps (Monday=0, Sunday=6)
    expected_days = np.array([0, 1, 2, 3, 4, 5, 6])

    # Calculate expected sin and cos based on the current implementation
    # The phase calculation uses day_week * (2π / 7)
    expected_phases = expected_days * (2 * np.pi / 7)
    expected_sin_dw = np.sin(expected_phases)
    expected_cos_dw = np.cos(expected_phases)

    # Assert that the calculated values match the expected values
    assert_array_almost_equal(sin_dw, expected_sin_dw, decimal=6)
    assert_array_almost_equal(cos_dw, expected_cos_dw, decimal=6)

    # In a proper cyclical encoding, cos(0) should be 1
    assert abs(cos_dw[0] - 1.0) < 0.0001


def test_time_cues_specific_days():
    """Test time_cues with specific known days of the week"""
    # Test with specific known dates
    test_dates = [
        # (date, expected_day_of_week)
        (datetime(2020, 1, 6, tzinfo=timezone.utc), 0),  # Monday
        (datetime(2020, 1, 7, tzinfo=timezone.utc), 1),  # Tuesday
        (datetime(2020, 1, 8, tzinfo=timezone.utc), 2),  # Wednesday
        (datetime(2020, 1, 9, tzinfo=timezone.utc), 3),  # Thursday
        (datetime(2020, 1, 10, tzinfo=timezone.utc), 4),  # Friday
        (datetime(2020, 1, 11, tzinfo=timezone.utc), 5),  # Saturday
        (datetime(2020, 1, 12, tzinfo=timezone.utc), 6),  # Sunday
    ]

    for date, expected_day in test_dates:
        timestamp = np.array([int(date.timestamp() * 1_000_000_000)])
        _, _, sin_dw, cos_dw, _, _, _, _, _ = time_cues(timestamp)

        # Calculate expected values
        expected_phase = expected_day * (2 * np.pi / 7)
        expected_sin = np.sin(expected_phase)
        expected_cos = np.cos(expected_phase)

        # Assert that calculated values match expected values
        assert_array_almost_equal(sin_dw, np.array([expected_sin]), decimal=6)
        assert_array_almost_equal(cos_dw, np.array([expected_cos]), decimal=6)


def test_time_cues_time_of_day():
    """Test that time of day features are correctly calculated"""
    # Create timestamps for different hours in a day
    base_date = datetime(2020, 1, 6, 0, 0, 0, tzinfo=timezone.utc)
    timestamps = np.array([
        int((base_date + timedelta(hours=h)).timestamp() * 1_000_000_000)
        for h in range(24)
    ])

    # Run the time_cues function
    sin_td, cos_td, _, _, _, _, _, _, _ = time_cues(timestamps)

    # Expected values for sin_td and cos_td
    expected_phases = np.array(range(24)) * (2 * np.pi / 24)
    expected_sin_td = np.sin(expected_phases)
    expected_cos_td = np.cos(expected_phases)

    # Assert that the calculated values match the expected values
    assert_array_almost_equal(sin_td, expected_sin_td, decimal=6)
    assert_array_almost_equal(cos_td, expected_cos_td, decimal=6)


def test_time_cues_session_flags():
    """Test that session flags are correctly set"""
    # Create timestamps for different hours in a day
    base_date = datetime(2020, 1, 6, 0, 0, 0, tzinfo=timezone.utc)
    timestamps = np.array([
        int((base_date + timedelta(hours=h)).timestamp() * 1_000_000_000)
        for h in range(24)
    ])

    # Run the time_cues function
    _, _, _, _, asia, eu, us, _, _ = time_cues(timestamps)

    # Expected values for session flags
    expected_asia = np.array([1 if 0 <= h < 8 else 0 for h in range(24)], dtype=np.bool_)
    expected_eu = np.array([1 if 7 <= h < 15 else 0 for h in range(24)], dtype=np.bool_)
    expected_us = np.array([1 if 13 <= h < 21 else 0 for h in range(24)], dtype=np.bool_)

    # Assert that the calculated values match the expected values
    assert_array_equal(asia, expected_asia)
    assert_array_equal(eu, expected_eu)
    assert_array_equal(us, expected_us)


def test_time_cues_transition_flags():
    """Test that transition flags are correctly set"""
    # Create timestamps for specific hours and minutes
    test_times = []
    base_date = datetime(2020, 1, 6, 0, 0, 0, tzinfo=timezone.utc)

    # Add every hour at 0 minutes
    for hour in range(24):
        test_times.append(base_date + timedelta(hours=hour, minutes=0))

    # Add also non-transition times (hours 0, 7, 13 but not at minute 0)
    test_times.append(base_date + timedelta(hours=0, minutes=1))
    test_times.append(base_date + timedelta(hours=7, minutes=1))
    test_times.append(base_date + timedelta(hours=13, minutes=1))

    timestamps = np.array([int(dt.timestamp() * 1_000_000_000) for dt in test_times])

    # Run the time_cues function
    _, _, _, _, _, _, _, trans, top_hr = time_cues(timestamps)

    # Expected values for transition flags
    expected_trans = np.zeros(len(test_times), dtype=np.bool_)
    expected_top_hr = np.zeros(len(test_times), dtype=np.bool_)

    # Set expected transition flags
    for i in range(24):
        if i in (0, 7, 13):
            expected_trans[i] = 1
        expected_top_hr[i] = 1

    # Assert that the calculated values match the expected values
    assert_array_equal(trans, expected_trans)
    assert_array_equal(top_hr, expected_top_hr)


def test_unix_epoch_day():
    """Test that Unix epoch (Jan 1, 1970) is correctly identified as Thursday"""
    # Unix epoch timestamp (Jan 1, 1970, 00:00:00 UTC)
    unix_epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
    timestamp = np.array([int(unix_epoch.timestamp() * 1_000_000_000)])

    # Run the time_cues function
    _, _, sin_dw, cos_dw, _, _, _, _, _ = time_cues(timestamp)

    # Thursday should be day 3 when Monday is 0
    expected_phase = 3 * (2 * np.pi / 7)
    expected_sin = np.sin(expected_phase)
    expected_cos = np.cos(expected_phase)

    # Assert that calculated values match expected values
    assert_array_almost_equal(sin_dw, np.array([expected_sin]), decimal=6)
    assert_array_almost_equal(cos_dw, np.array([expected_cos]), decimal=6)


if __name__ == '__main__':
    pytest.main()
