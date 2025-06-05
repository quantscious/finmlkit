"""
Feature transform wrapper for financial time series data.
"""
from .base import SISOTransform, SIMOTransform, MISOTransform, BaseTransform
from .core.utils import comp_lagged_returns, comp_zscore, comp_burst_ratio, pct_change
from .core.volatility import ewmst, realized_vol, bollinger_percent_b, parkinson_range
from .core.volume import comp_flow_acceleration
from .core.reversion import vwap_distance
from .core.time import time_cues
from .core.ma import ewma, sma
from .core.momentum import roc, rsi_wilder, stoch_k
from typing import Union
from finmlkit.utils.log import get_logger
import pandas as pd
import numpy as np

logger = get_logger(__name__)


class Identity(BaseTransform):
    """
    Returns the identity transform of a selected column
    """
    def __init__(self, input_col: str = "close"):
        """
        Identity transform that returns the input column as is.

        :param input_col: If DataFrame is passed, this is the column name to return.
        """
        assert isinstance(input_col, str), "Input column must be a string."
        super().__init__(input_col, input_col)

    def __call__(self, df: pd.DataFrame, *, backend="nb") -> pd.Series:
        """
        Returns the selected col as a series.

        :param df: Input DataFrame.
        :param backend: Ignored for this transform.
        :return: The input DataFrame with the specified column.
        """
        self._validate_input(df)

        return df[self.requires[0]]

    def _validate_input(self, x: pd.DataFrame) -> bool:
        if self.requires[0] not in x.columns:
            raise ValueError(f"Input DataFrame must contain the column '{self.requires[0]}'.")
        return True

    def output_name(self) -> str:
        """
        Returns the name of the output column.
        """
        return self.produces[0]


class Lag(SISOTransform):
    """
    Implements lagged values of a time series.
    """

    def __init__(self, periods: int = 1, input_col: str = "close"):
        """
        Compute lagged values over the specified number of periods.

        :param input_col: If DataFrame is passed, this is the column name to compute lags on.
        :param periods: The lag period.
        """
        super().__init__(input_col, f"lag{periods}")
        self.periods = periods

    def _pd(self, x):
        series = x[self.requires[0]]
        outp = series.shift(self.periods)
        outp.name = self.output_name

        return outp

    def _nb(self, x: Union[pd.DataFrame, pd.Series]) -> pd.Series:
        logger.info(f"Fall back to pandas for {self.__class__.__name__}")
        return self._pd(x)

class ReturnT(SISOTransform):
    """
    Calculates the lagged returns of a time series using a specified period defined in seconds.
    Works for irregular time series too.
    """
    def __init__(self, window_sec: float = 1e-6, is_log: bool = False, input_col: str = "close"):
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

    def _pd(self, x):
        logger.info(f"Fall back to numba for {self.__class__.__name__}")
        return self._nb(x)

    def _nb(self, x: pd.DataFrame) -> pd.Series:
        input_arr = self._prepare_input_nb(x)
        ts = self._get_timestamps(x)
        result = comp_lagged_returns(ts, input_arr, self.window_sec, self.is_log)

        return self._prepare_output_nb(x.index, result)


class Return(SISOTransform):
    """
    Implements lagged return
    """
    def __init__(self, periods: int = 1, input_col: str = "close", is_log: bool = False):
        """
        Compute lagged returns over the specified number of periods.

        :param input_col: If DataFrame is passed, this is the column name to compute returns on.
        :param periods: The lag period.
        :param is_log: If True, compute log returns. Otherwise, compute simple returns.
        """
        super().__init__(input_col, f"ret{periods}")
        self.periods = periods
        self.is_log = is_log

    def _pd(self, x):
        series = x[self.requires[0]]
        series_lagged = series.shift(self.periods)
        if self.is_log:
            # Calculate log returns
            outp = np.log(series/ series_lagged)
        else:
            # Calculate simple returns
            outp = series / series_lagged - 1.0

        outp = pd.Series(outp, index=series.index, name=self.output_name)

        return outp

    def _nb(self, x: Union[pd.DataFrame, pd.Series]) -> pd.Series:
        logger.info(f"Fall back to pandas for {self.__class__.__name__}")
        return self._pd(x)


class ROC(SISOTransform):
    """
    Computes the Rate of Change (ROC) of a time series.
    """
    def __init__(self, periods: int, input_col: str = "close"):
        """
        Compute the Rate of Change (ROC) of a time series.

        :param input_col: If DataFrame is passed, this is the column name to compute returns on.
        :param periods: The lag periods for the ROC calculation.
        """
        super().__init__(input_col, f"roc{periods}")
        self.periods = periods

    def _pd(self, x):
        logger.info(f"Fall back to numba for {self.__class__.__name__}")
        return self._nb(x)

    def _nb(self, x: pd.DataFrame) -> pd.Series:
        input_arr = self._prepare_input_nb(x)
        result = roc(input_arr, self.periods)

        return self._prepare_output_nb(x.index, result)


class PctChange(SISOTransform):
    """
    Computes the percentage change of a time series using a specified lag.
    """
    def __init__(self, window: int, input_col: str = "close"):
        """
        Calculate the percentage change of a signal with a specified lag.

        :param input_col: If DataFrame is passed, this is the column name to compute returns on.
        :param periods: The lag period.
        :return: The percentage change of the signal.
        """
        super().__init__(input_col, f"pctc{window}")
        self.periods = window

    def _pd(self, x: pd.DataFrame) -> pd.Series:
        series = x[self.requires[0]]
        return series.pct_change(self.periods)

    def _nb(self, x: Union[pd.DataFrame, pd.Series]) -> pd.Series:
        input_arr = self._prepare_input_nb(x)
        result = pct_change(input_arr, self.periods)

        return self._prepare_output_nb(x.index, result)


class RSIWilder(SISOTransform):
    """
    Computes the Relative Strength Index (RSI) of a time series using Wilder's smoothing method.
    """

    def __init__(self, window: int = 14, input_col: str = "close"):
        """
        Compute the Relative Strength Index (RSI) of a time series.

        :param input_col: If DataFrame is passed, this is the column name to compute RSI on.
        :param window: Window size for the RSI calculation.
        """
        super().__init__(input_col, f"rsiw{window}")
        self.window = window

    def _pd(self, x):
        # Get the input series
        series = x[self.requires[0]]

        # Calculate price changes
        delta = series.diff()

        # Split gains and losses
        gains = delta.copy()
        losses = delta.copy()
        gains[gains < 0] = 0
        losses[losses > 0] = 0
        losses = -losses  # Make losses positive

        # Initialize results with NaNs
        rsi = pd.Series(np.nan, index=series.index)

        # First average is simple average
        first_avg_gain = gains.iloc[1:self.window + 1].mean()
        first_avg_loss = losses.iloc[1:self.window + 1].mean()

        # Use Wilder's smoothing method
        avg_gain = first_avg_gain
        avg_loss = first_avg_loss

        # Calculate first RSI value
        if first_avg_loss != 0:
            rs = avg_gain / avg_loss
        else:
            rs = float('inf')  # Avoid division by zero

        rsi.iloc[self.window] = 100 - (100 / (1 + rs)) if rs != float('inf') else 100

        # Subsequent averages use Wilder's smoothing
        for i in range(self.window + 1, len(series)):
            avg_gain = ((self.window - 1) * avg_gain + gains.iloc[i]) / self.window
            avg_loss = ((self.window - 1) * avg_loss + losses.iloc[i]) / self.window

            if avg_loss != 0:
                rs = avg_gain / avg_loss
            else:
                rs = float('inf')

            rsi.iloc[i] = 100 - (100 / (1 + rs)) if rs != float('inf') else 100

        rsi.name = self.output_name
        return rsi

    def _nb(self, x: Union[pd.DataFrame, pd.Series]) -> pd.Series:
        input_arr = self._prepare_input_nb(x)
        result = rsi_wilder(input_arr, self.window)

        return self._prepare_output_nb(x.index, result)


class StochK(MISOTransform):
    """
    Computes the Stochastic Oscillator %K of a time series.
    """
    def __init__(self, length: int = 14, input_cols: list[str] = None):
        """
        Compute the Stochastic Oscillator %K of a time series.

        :param input_cols: If DataFrame is passed, this is the column names to compute %K on. Default is ["high", "low", "close"].
        :param length: The lookback period for the stochastic calculation.
        """
        if input_cols is None:
            input_cols = ["high", "low", "close"]

        super().__init__(input_cols, f"stochk{length}")
        self.length = length

    def _pd(self, x):
        logger.info(f"Fall back to numba for {self.__class__.__name__}")
        return self._nb(x)

    def _nb(self, x: pd.DataFrame) -> pd.Series:
        input_dict = self._prepare_input_nb(x)
        high = input_dict[self.requires[0]]
        low = input_dict[self.requires[1]]
        close = input_dict[self.requires[2]]

        result = stoch_k(high, low, close, self.length)

        return self._prepare_output_nb(x.index, result)


class EWMST(SISOTransform):
    """
    Computes the exponentially weighted moving standard deviation of a time series.
    """
    def __init__(self, half_life_sec: float, input_col: str):
        """
        Compute the exponentially weighted moving standard deviation.

        :param input_col: If DataFrame is passed, this is the column name to compute returns on.
        :param half_life_sec: Period for the moving standard deviation.
        """
        super().__init__(input_col, f"ewms{half_life_sec}s")
        self.half_life_sec = half_life_sec

    def _pd(self, x):
        logger.info(f"Fall back to numba for {self.__class__.__name__}")
        return self._nb(x)

    def _nb(self, x: pd.DataFrame) -> pd.Series:
        input_arr = self._prepare_input_nb(x)
        ts = self._get_timestamps(x)
        result = ewmst(ts, input_arr, self.half_life_sec)

        return self._prepare_output_nb(x.index, result)


class ZScore(SISOTransform):
    """
    Computes the z-score of a time series using a rolling window.
    """
    def __init__(self,  window: int, input_col: str, ddof: int = 0):
        """
        Compute the z-score of a time series using a rolling window.

        :param input_col: If DataFrame is passed, this is the column name to compute returns on.
        :param window: Window size for the rolling calculation.
        :param ddof: Delta degrees of freedom for standard deviation calculation.
        """
        super().__init__(input_col, f"z{window}")
        self.window = window
        self.ddof = ddof

    def _pd(self, x):
        logger.info(f"Fall back to numba for {self.__class__.__name__}")
        return self._nb(x)

    def _nb(self, x: pd.DataFrame) -> pd.Series:
        input_arr = self._prepare_input_nb(x)
        result = comp_zscore(input_arr, self.window, self.ddof)

        return self._prepare_output_nb(x.index, result)


class BurstRatio(SISOTransform):
    """
    Computes the burst ratio of a time series using a rolling window.
    """
    def __init__(self, window: int, input_col: str):
        """
        Compute the burst ratio of a time series.

        :param input_col: If DataFrame is passed, this is the column name to compute returns on.
        :param window: Window size for the rolling calculation.
        :return: series of burst ratios (same size as input)
        """
        super().__init__(input_col, f"burst{window}")
        self.window = window

    def _pd(self, x):
        logger.info(f"Fall back to numba for {self.__class__.__name__}")
        return self._nb(x)

    def _nb(self, x: pd.DataFrame) -> pd.Series:
        input_arr = self._prepare_input_nb(x)
        result = comp_burst_ratio(input_arr, self.window)

        return self._prepare_output_nb(x.index, result)


class VWAPDistance(MISOTransform):
    """
    Computes the distance of the current price from the VWAP (Volume Weighted Average Price).
    """
    def __init__(self, periods: int, is_log: bool = False, input_cols: str = None):
        """
        Calculate the distance of the current price from the VWAP (Volume Weighted Average Price).

        :param input_cols: close and volume columns. Default is ["close", "volume"] when None.
        :param periods: Number of periods to calculate VWAP.
        :param is_log: If True, calculate log distance.
        """
        if input_cols is None:
            input_cols = ["close", "volume"]

        super().__init__(input_cols, f"vwapd{periods}")
        self.periods = periods
        self.is_log = is_log

    def _pd(self, x):
        logger.info(f"Fall back to numba for {self.__class__.__name__}")
        return self._nb(x)

    def _nb(self, x: Union[pd.DataFrame, pd.Series]) -> pd.Series:
        input_dict = self._prepare_input_nb(x)
        close = input_dict[self.requires[0]]
        volume = input_dict[self.requires[1]]

        result = vwap_distance(close, volume, self.periods, self.is_log)

        return self._prepare_output_nb(x.index, result)


class TimeCues(SIMOTransform):
    """
    Computes time-based features for a time series.
    """
    def __init__(self, input_col: str = "close"):
        """
        Compute time-based features for a time series.

        :param input_col: If DataFrame is passed, this is the column name to compute returns on.
        """
        produces=["sin_td", "cos_td", "dow", "asia", "eu", "us", "sess_x", "top_hr"]
        super().__init__(input_col, produces)

    def _pd(self, x):
        logger.info(f"Fall back to numba for {self.__class__.__name__}")
        return self._nb(x)

    def _nb(self, x: pd.DataFrame) -> tuple[pd.Series, ...]:
        ts = self._get_timestamps(x)
        result = time_cues(ts)
        result = list(self._prepare_output_nb(x.index, result))

        # set dow as categorical column
        result[2] = result[2].astype("category")

        return tuple(result)

    @property
    def output_name(self):
        return self.produces



class RealizedVolatility(SISOTransform):
    """
    Computes the realized volatility of a time series.
    """
    def __init__(self, window: int, is_sample=False, input_col: str = "ret"):
        """
        Compute the realised volatility of a time series.

        :param input_col: If DataFrame is passed, this is the column name to compute returns on.
        :param is_sample: If True, use sample standard deviation. If False, use population standard deviation.
        :param window: Window size for the rolling calculation.
        """
        super().__init__(input_col, f"rv{window}")
        self.window = window
        self.is_sample = is_sample

    def _pd(self, x):
        # Get input series
        series = x[self.requires[0]]

        # Create result series filled with NaNs
        result = pd.Series(np.nan, index=series.index, name=self.output_name)

        # Calculate only from window-1 onward (matching numba implementation)
        for i in range(self.window - 1, len(series)):
            window_data = series.iloc[i - self.window + 1:i + 1]
            valid_count = window_data.count()

            if valid_count > 1:
                divisor = (valid_count - 1) if self.is_sample else valid_count
                result.iloc[i] = np.sqrt((window_data ** 2).sum(skipna=True) / divisor)


        result.name = self.output_name

        return result

    def _nb(self, x: Union[pd.DataFrame, pd.Series]) -> pd.Series:
        input_arr = self._prepare_input_nb(x)
        result = realized_vol(input_arr.astype(np.float64), self.window, self.is_sample)

        return self._prepare_output_nb(x.index, result)


class BollingerPercentB(SISOTransform):
    """
    Computes the Bollinger Percent B of a time series.
    """
    def __init__(self, window: int, num_std: float= 2., input_col: str = "close"):
        """
        Compute the Bollinger Percent B of a time series.

        :param input_col: If DataFrame is passed, this is the column name to compute returns on.
        :param num_std: Number of standard deviations for the Bollinger Bands.
        :param window: Window size for the rolling calculation.
        """
        super().__init__(input_col, f"bollb{window}")
        self.window = window
        self.num_std = num_std

    def _pd(self, x):
        logger.info(f"Fall back to numba for {self.__class__.__name__}")
        return self._nb(x)

    def _nb(self, x: Union[pd.DataFrame, pd.Series]) -> pd.Series:
        input_arr = self._prepare_input_nb(x)
        result = bollinger_percent_b(input_arr, self.window, self.num_std)

        return self._prepare_output_nb(x.index, result)


class ParkinsonRange(MISOTransform):
    """
    Computes the Parkinson range of a time series.
    """
    def __init__(self, input_cols=None):
        """
        Compute the Parkinson range of a time series.

        :param input_cols: High and Low columns. If None defaults to ["high", "low"].
        """
        if input_cols is None:
            input_cols = ["high", "low"]
        super().__init__(input_cols, f"parkrange")

    def _pd(self, x):
        logger.info(f"Fall back to numba for {self.__class__.__name__}")
        return self._nb(x)

    def _nb(self, x: Union[pd.DataFrame, pd.Series]) -> pd.Series:
        input_dict = self._prepare_input_nb(x)
        high = input_dict[self.requires[0]]
        low = input_dict[self.requires[1]]

        result = parkinson_range(high, low)

        return self._prepare_output_nb(x.index, result)


class SMA(SISOTransform):
    """
    Computes the Simple Moving Average (SMA) of a time series.
    """
    def __init__(self, window: int, input_col: str = "x"):
        """
        Compute the Simple Moving Average (SMA) of a time series.

        :param input_col: If DataFrame is passed, this is the column name to compute returns on.
        :param window: Window size for the rolling calculation.
        """
        super().__init__(input_col, f"sma{window}")
        self.window = window

    def _pd(self, x):
        series = x[self.requires[0]]
        outp = series.rolling(window=self.window).mean()
        outp.name = self.produces

        return outp

    def _nb(self, x: Union[pd.DataFrame, pd.Series]) -> pd.Series:
        input_arr = self._prepare_input_nb(x)
        result = sma(input_arr, self.window)

        return self._prepare_output_nb(x.index, result)


class EWMA(SISOTransform):
    """
    Computes the Exponentially Weighted Moving Average (EWMA) of a time series.
    """
    def __init__(self, span: int, input_col: str = None):
        """
        Compute the Exponentially Weighted Moving Average (EWMA) of a time series.

        :param input_col: If DataFrame is passed, this is the column name to compute returns on.
        :param span: The decay window, or 'span'. Determines how many past points meaningfully impact the EWMA value.
        """
        super().__init__(input_col, f"ewma{span}")
        self.span = span

    def _pd(self, x):
        input_series = x[self.requires[0]]
        outp = input_series.ewm(span=self.span).mean()
        outp.name = self.output_name

        return outp

    def _nb(self, x: Union[pd.DataFrame, pd.Series]) -> pd.Series:
        input_arr = self._prepare_input_nb(x)
        result = ewma(input_arr, self.span)

        return self._prepare_output_nb(x.index, result)


class FlowAcceleration(SISOTransform):
    """
    Computes the Flow Acceleration of a time series.
    """
    def __init__(self, window: int, recent_periods, input_col: str = "volume"):
        """
        Compute the Flow Acceleration of a time series.

        :param input_col: If DataFrame is passed, this is the column name to compute returns on.
        :param window: Window size for the rolling calculation.
        """
        super().__init__(input_col, f"flowacc_{window}_{recent_periods}")
        self.window = window
        self.recent_periods = recent_periods

    def _pd(self, x):
        logger.info(f"Fall back to numba for {self.__class__.__name__}")
        return self._nb(x)

    def _nb(self, x: pd.DataFrame) -> pd.Series:
        input_arr = self._prepare_input_nb(x)
        result = comp_flow_acceleration(input_arr, self.window, self.recent_periods)

        return self._prepare_output_nb(x.index, result)

