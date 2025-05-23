"""
Feature transform wrapper for financial time series data.
"""
from .base import BaseTransform
from .core.utils import comp_lagged_returns, comp_zscore, comp_burst_ratio, pct_change
from .core.volatility import ewmst
from .core.reversion import vwap_distance
from typing import Union
from utils.log import get_logger
import pandas as pd

logger = get_logger(__name__)


class ReturnT(BaseTransform):
    """
    Calculates the lagged returns of a time series using a specified period defined in seconds.
    Works for irregular time series too.
    """
    def __init__(self, window_sec: float = 1e-6, is_log: bool = False, input_col: str = None):
        """
        Compute lagged returns over the specified time window. Works for unregular time series too.

        :param input_col: If DataFrame is passed, this is the column name to compute returns on.
        :param window_sec: Period in seconds for the lagged returns. Set it to a small value (e.g. 1e-6) for 1 sample lag.
        :param is_log: If True, compute log returns. Otherwise, compute simple returns.
        """
        output_col = f"ret{window_sec}s" if window_sec > 1e-6 else "ret1"
        super().__init__(input_col, output_col)
        self.window_sec = window_sec
        self.is_log = is_log

    def transform_pandas(self, x):
        logger.info(f"Fall back to numba for {self.__class__.__name__}")
        return self.transform_numba(x)

    def transform_numba(self, x: Union[pd.DataFrame, pd.Series]) -> pd.Series:
        input_arr = self.get_input_arrays(x)
        ts = self.get_timestamps(x)
        result = comp_lagged_returns(ts, input_arr, self.window_sec, self.is_log)

        return self.to_series(x, result)


class EWMST(BaseTransform):
    """
    Computes the exponentially weighted moving standard deviation of a time series.
    """
    def __init__(self, half_life_sec: float, input_col: str = None):
        """
        Compute the exponentially weighted moving standard deviation.

        :param input_col: If DataFrame is passed, this is the column name to compute returns on.
        :param half_life_sec: Period for the moving standard deviation.
        """
        super().__init__(input_col, f"ewms{half_life_sec}s")
        self.half_life_sec = half_life_sec

    def transform_pandas(self, x):
        logger.info(f"Fall back to numba for {self.__class__.__name__}")
        return self.transform_numba(x)

    def transform_numba(self, x: Union[pd.DataFrame, pd.Series]) -> pd.Series:
        input_arr = self.get_input_arrays(x)
        ts = self.get_timestamps(x)
        result = ewmst(ts, input_arr, self.half_life_sec)

        return self.to_series(x, result)


class ZScore(BaseTransform):
    """
    Computes the z-score of a time series using a rolling window.
    """
    def __init__(self,  window: int, ddof: int = 0, input_col: str = None):
        """
        Compute the z-score of a time series using a rolling window.

        :param input_col: If DataFrame is passed, this is the column name to compute returns on.
        :param window: Window size for the rolling calculation.
        :param ddof: Delta degrees of freedom for standard deviation calculation.
        """
        super().__init__(input_col, f"z{window}")
        self.window = window
        self.ddof = ddof

    def transform_pandas(self, x):
        logger.info(f"Fall back to numba for {self.__class__.__name__}")
        return self.transform_numba(x)

    def transform_numba(self, x: Union[pd.DataFrame, pd.Series]) -> pd.Series:
        input_arr = self.get_input_arrays(x)
        result = comp_zscore(input_arr, self.window, self.ddof)

        return self.to_series(x, result)


class BurstRatio(BaseTransform):
    """
    Computes the burst ratio of a time series using a rolling window.
    """
    def __init__(self, window: int, input_col: str = None):
        """
        Compute the burst ratio of a time series.

        :param input_col: If DataFrame is passed, this is the column name to compute returns on.
        :param window: Window size for the rolling calculation.
        :return: series of burst ratios (same size as input)
        """
        super().__init__(input_col, f"burst{window}")
        self.window = window

    def transform_pandas(self, x):
        logger.info(f"Fall back to numba for {self.__class__.__name__}")
        return self.transform_numba(x)

    def transform_numba(self, x: Union[pd.DataFrame, pd.Series]) -> pd.Series:
        input_arr = self.get_input_arrays(x)
        result = comp_burst_ratio(input_arr, self.window)

        return self.to_series(x, result)


class PctChange(BaseTransform):
    """
    Computes the percentage change of a time series using a specified lag.
    """
    def __init__(self, window: int, input_col: str = None):
        """
        Calculate the percentage change of a signal with a specified lag.

        :param input_col: If DataFrame is passed, this is the column name to compute returns on.
        :param periods: The lag period.
        :return: The percentage change of the signal.
        """
        super().__init__(input_col, f"pctc{window}")
        self.periods = window

    def transform_pandas(self, x: Union[pd.DataFrame, pd.Series]) -> pd.Series:
        series = self.get_series(x)
        return series.pct_change(self.periods)

    def transform_numba(self, x: Union[pd.DataFrame, pd.Series]) -> pd.Series:
        input_arr = self.get_input_arrays(x)
        result = pct_change(input_arr, self.periods)

        return self.to_series(x, result)


class VWAPDistance(BaseTransform):
    """
    Computes the distance of the current price from the VWAP (Volume Weighted Average Price).
    """
    def __init__(self, n_periods: int, is_log: bool = False, input_cols: str = None):
        """
        Calculate the distance of the current price from the VWAP (Volume Weighted Average Price).

        :param input_cols: close and volume columns. Default is ["close", "volume"] when None.
        :param n_periods: Number of periods to calculate VWAP.
        :param is_log: If True, calculate log distance.
        """
        if input_cols is None:
            input_cols = ["close", "volume"]

        super().__init__(input_cols, f"vwapd{n_periods}")
        self.n_periods = n_periods
        self.is_log = is_log

    def transform_pandas(self, x):
        logger.info(f"Fall back to numba for {self.__class__.__name__}")
        return self.transform_numba(x)

    def transform_numba(self, x: pd.DataFrame) -> pd.Series:
        input_dict = self.get_input_arrays(x)
        close = input_dict[self.requires[0]]
        volume = input_dict[self.requires[1]]

        result = vwap_distance(close, volume)

        return self.to_series(x, result)