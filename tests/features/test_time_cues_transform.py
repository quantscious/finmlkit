"""
Tests for the TimeCues transform class with the updated time_cues function
"""
import pandas as pd
import pytest
import numpy as np
from datetime import datetime, timezone, timedelta
from pandas.testing import assert_series_equal, assert_frame_equal

from finmlkit.feature.transforms import TimeCues


def create_test_dataframe():
    """Create a test DataFrame with a date range index"""
    # Create a week of timestamps starting from Monday
    start_date = datetime(2020, 1, 6, tzinfo=timezone.utc)
    dates = pd.date_range(start=start_date, periods=7, freq='D')

    # Create test DataFrame with a dummy close column
    df = pd.DataFrame(index=dates)
    df['close'] = [100.0, 101.0, 102.0, 101.5, 103.0, 102.5, 104.0]

    return df


def test_time_cues_transform():
    """Test that the TimeCues transform correctly extracts time features"""
    # Create test data
    df = create_test_dataframe()

    # Apply the transform
    time_cues = TimeCues(input_col="close")
    result = time_cues(df)

    # Unpack the result tuple into separate series
    sin_td, cos_td, dow, asia, eu, us, sess_x, top_hr = result

    # Check that we get the expected number of output series
    assert len(result) == len(time_cues.produces), "TimeCues transform should return the correct number of series"

    # Check the day of week values for each day
    expected_dow = pd.Series(
        [0, 1, 2, 3, 4, 5, 6],
        index=df.index,
        name='dow'
    ).astype('category')

    # Check dow is categorical
    assert dow.dtype == 'category', "The dow column should be categorical"

    # Compare values - must convert to integers since category comparisons might fail
    assert all(dow.astype(int).values == expected_dow.astype(int).values), \
        "Day of week values should match expected values"

    # Check other columns exist
    assert sin_td.name == 'sin_td'
    assert cos_td.name == 'cos_td'
    assert asia.name == 'asia'
    assert eu.name == 'eu'
    assert us.name == 'us'
    assert sess_x.name == 'sess_x'
    assert top_hr.name == 'top_hr'


def test_time_cues_transform_output_names():
    """Test that the TimeCues transform produces columns with the correct names"""
    # Create test data
    df = create_test_dataframe()

    # Create the transform
    time_cues = TimeCues(input_col="close")

    # Apply the transform
    result = time_cues(df)
    _, _, dow, _, _, _, _, _ = result

    # Check output names
    expected_names = ["sin_td", "cos_td", "dow", "asia", "eu", "us", "sess_x", "top_hr"]
    assert time_cues.produces == expected_names, "TimeCues output names should match expected names"

    assert dow.dtype == 'category', "dow column should be categorical type"



if __name__ == '__main__':
    pytest.main()
