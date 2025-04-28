"""
Pandas syntactic sugar for accessing indicators.
"""
import pandas as pd
import numpy as np
from pandas.api.extensions import register_dataframe_accessor


@register_dataframe_accessor("fmk.volty")
class VolatilityAccessor:
    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    def standard(self, timestamp_col='timestamp', close_col='close',
                 return_window_sec=60.0, lookback=100, result_col='volatility'):
        """
        Implements a simple volatility estimator using an exponentially weighted rolling window standard deviation on lagged returns
         and add it as a new column to the dataframe.

        :param timestamp_col: Column name containing timestamps (in nanoseconds)
        :param close_col: Column name containing close prices
        :param return_window_sec: The lag window size in seconds to compute returns
        :param lookback: Number of points for EWM std lookback
        :param result_col: Name of the column to store the volatility results
        :return: DataFrame with added volatility column
        """
        from .volatility import standard_volatility_estimator

        df = self._obj.copy()

        if timestamp_col not in df.columns or close_col not in df.columns:
            raise ValueError(f"Required columns not found: {timestamp_col} or {close_col}")

        timestamps = df[timestamp_col].to_numpy(dtype=np.int64)
        close_prices = df[close_col].to_numpy(dtype=np.float64)

        vol = standard_volatility_estimator(
            timestamps, close_prices, return_window_sec, lookback
        )

        df[result_col] = vol
        return df


@register_dataframe_accessor("fmk")
class EWMAccessor:
    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    def ewma(self, column="close", span=100, result_col=None):
        """
        Calculate Exponentially Weighted Moving Average for a column.

        :param column: str, column name to apply EWMA to, defaults to "close"
        :param span: int, the decay window or 'span' for the EWMA calculation, defaults to 100
        :param result_col: str or None, name of the column to store results, defaults to "{column}_ewma_{span}"
        :returns: pd.DataFrame with the EWMA column added
        :raises ValueError: If column is not found in the DataFrame
        """
        from .ma import ewma as ewma_nb

        df = self._obj.copy()

        # Check if column exists
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame")

        # Set result column name
        if result_col is None:
            result_col = f"{column}_ewma_{span}"

        # Calculate EWMA
        df[result_col] = ewma_nb(df[column].to_numpy(dtype=np.float64), span)

        return df

    def ewms(self, column="close", span=10, result_col=None):
        """
        Calculate Exponentially Weighted Moving Standard Deviation for a column.

        :param column: str, column name to apply EWMS to, defaults to "close"
        :param span: int, the decay window or 'span' for the EWMS calculation, defaults to 10
        :param result_col: str or None, name of the column to store results, defaults to "{column}_ewms_{span}"
        :returns: pd.DataFrame with the EWMS column added
        :raises ValueError: If column is not found in the DataFrame
        """
        from .volatility import ewms as ewms_func

        df = self._obj.copy()

        # Check if column exists
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame")

        # Set result column name
        if result_col is None:
            result_col = f"{column}_ewms_{span}"

        # Calculate EWMS
        df[result_col] = ewms_func(df[column].to_numpy(dtype=np.float64), span)

        return df
