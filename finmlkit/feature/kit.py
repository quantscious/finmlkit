from .ma import ewma, sma
from .volatility import ewmst, ewms, realised_vol, bollinger_percent_b, true_range
from .structural_break.cusum import cusum_test_rolling
from .reversion import vwap_distance
from .utils import comp_lagged_returns, comp_zscore, comp_burst_ratio
from finmlkit.utils.log import get_logger
import pandas as pd
import numpy as np
from typing import Any, Callable, Sequence

logger = get_logger(__name__)


class FeatureBuilder:
    """
    A chainable builder for creating features from a specific column.
    """

    def __init__(self, df: pd.DataFrame, source_col: str, timestamps: np.ndarray, ohlcv_cols: dict = None):
        """
        Initialize a feature builder for a specific column.

        :param df: Dataframe containing the data
        :param source_col: The column to build features from
        :param timestamps: Array of timestamps (int64)
        :param ohlcv_cols: Dict mapping standard OHLCV names to actual column names
        """
        self.df = df
        self.source_col = source_col
        self.current_col = source_col
        self.timestamps = timestamps
        self.intermediate_cols = set()

        # Default OHLCV column mapping
        self.ohlcv_cols = {
            'open': 'open', 'high': 'high', 'low': 'low',
            'close': 'close', 'volume': 'volume'
        }
        # Override with user-specified mappings if provided
        if ohlcv_cols:
            self.ohlcv_cols.update(ohlcv_cols)

    def _add_feature(self, feature_func: Callable, feature_name: str, **kwargs) -> 'FeatureBuilder':
        """
        Generic method to apply a feature function and manage columns.

        :param feature_func: Feature calculation function to apply
        :param out_col: Output column name (internally defined, not user-exposed)
        :param kwargs: Parameters to pass to the feature function
        :return: Self for method chaining
        """
        # Apply the feature function
        result = feature_func(**kwargs)
        if isinstance(result, (tuple, list)):
            for i, res in enumerate(result):
                col_name = f"{feature_name}_{i}"
                self.df[col_name] = res
                self.intermediate_cols.add(col_name)
        else:
            # Add the new column and mark as intermediate
            self.df[feature_name] = result
            self.intermediate_cols.add(feature_name)

        # Always drop intermediate columns except the source column
        if self.current_col != self.source_col and self.current_col in self.intermediate_cols:
            self.df.drop(columns=self.current_col, inplace=True)

        # Update current column
        self.current_col = feature_name
        return self

    def _add_ohlcv_feature(self, feature_func: Callable, required_cols: list,
                           feature_name: str, **kwargs) -> 'FeatureBuilder':
        """
        Helper method to apply OHLCV/candle features.

        :param feature_func: Feature calculation function to apply
        :param required_cols: List of required OHLCV column types
        :param feature_name: Base name for the feature column
        :param kwargs: Additional parameters to pass to feature function
        :return: Self for method chaining
        """
        # Map column types to actual DataFrame column names
        col_mapping = {}
        for col_type in required_cols:
            col_name = self.ohlcv_cols.get(col_type, col_type)
            if col_name not in self.df.columns:
                raise ValueError(f"{col_type.capitalize()} column '{col_name}' not found in dataframe")
            col_mapping[col_type] = col_name

        # Generate output column name
        param_value = next(iter(kwargs.values()), "")
        param_suffix = f"{param_value}" if param_value else ""
        out_col = f"{feature_name}{param_suffix}"

        # Extract DataFrame columns and calculate feature
        input_arrays = [self.df[col_mapping[col]].values for col in required_cols]
        self.df[out_col] = feature_func(*input_arrays, **kwargs)

        # Drop intermediate columns except the source column
        if self.current_col != self.source_col and self.current_col in self.intermediate_cols:
            self.df.drop(columns=self.current_col, inplace=True)

        # Update state
        self.intermediate_cols.add(out_col)
        self.current_col = out_col

        return self

    def res(self) -> str:
        """Get the final column name of the feature chain."""
        return self.current_col

    def sma(self, window: int) -> 'FeatureBuilder':
        """
        Apply simple moving average (SMA) with specified window.
        :param window: Window size for the SMA calculation.
        :return: Self for method chaining
        """
        return self._add_feature(
            lambda: sma(self.df[self.current_col].values, window),
            feature_name=f"sma{window}",
            window=window
        )

    def ewma(self, span: int) -> 'FeatureBuilder':
        """
        Apply EWMA with specified span.
        :param span: Span for the EWMA calculation.
        :return: Self for method chaining
        """
        return self._add_feature(
            lambda: ewma(self.df[self.current_col].values, span),
            feature_name=f"ewma{span}",
            span=span
        )

    def ewms(self, span: int) -> 'FeatureBuilder':
        """
        Apply exponentially weighted moving standard deviation (EWMS) with specified span.
        :param span: Span for the EWMS calculation.
        :return: Self for method chaining
        """
        return self._add_feature(
            lambda: ewms(self.df[self.current_col].values, span),
            feature_name=f"ewms{span}",
            span=span
        )

    def ewmst(self, half_life_sec: float) -> 'FeatureBuilder':
        """
        Apply a temporal exponentially weighted moving standard deviation (EWMS) with specified half-life.
        :param half_life_sec: Half-life in seconds for the EWMS calculation.
        :return: Self for method chaining
        """
        return self._add_feature(
            lambda: ewmst(self.timestamps, self.df[self.current_col].values, half_life_sec),
            feature_name=f"ewmst{half_life_sec}s",
            half_life_sec=half_life_sec
        )

    def ret(self, return_window_sec: float, is_log: bool = True) -> 'FeatureBuilder':
        """
        Compute lagged returns over the specified time window. Works for unregular time series too.

        :param return_window_sec: Return window in seconds. Set to a small value (e.g. 1e-6) for 1 sample lag.
        :param is_log: If True, compute log returns instead of simple returns.
        :return: Self for method chaining
        """
        return self._add_feature(
            lambda: comp_lagged_returns(self.timestamps, self.df[self.current_col].values, return_window_sec, is_log),
            feature_name=f"ret{return_window_sec}s" if return_window_sec > 1e-6 else "ret1",
            return_window_sec=return_window_sec,
            is_log=is_log
        )

    def zscore(self, window: int, ddof: int = 0) -> 'FeatureBuilder':
        """
        Compute the z-score of the current column over a specified window.

        :param window: Window size for the z-score calculation.
        :param ddof: Delta degrees of freedom for standard deviation calculation.
        :return: Self for method chaining
        """
        return self._add_feature(
            lambda: comp_zscore(self.df[self.current_col].values, window, ddof),
            feature_name=f"z{window}",
            window=window,
            ddof=ddof
        )

    def burst_ratio(self, window: int = 20) -> 'FeatureBuilder':
        """
        Compute the burst ratio of the current column over a specified window.

        :param window: Window size for the burst ratio calculation. Default is 20.
        :return: Self for method chaining
        """
        return self._add_feature(
            lambda: comp_burst_ratio(self.df[self.current_col].values, window),
            feature_name=f"burst{window}",
            window=window
        )

    def lag(self, periods: int = 1) -> 'FeatureBuilder':
        """
        Create a lagged version of the current column.

        :param periods: Number of periods to lag the series. Default is 1.
        :return: Self for method chaining
        """
        return self._add_feature(
            lambda: self.df[self.current_col].shift(periods),
            feature_name=f"lag{periods}",
            n_periods=periods
        )

    def rvola(self, window: int, is_sample=False) -> 'FeatureBuilder':
        """
        Compute realised volatility over the specified period for returns.

        :param window: Number of samples to compute realised volatility over.
        :param is_sample: To calculate sample or population volatility. Default is False.
        :return: Self for method chaining
        """
        # Check if the current column is a return series
        if "_ret" not in self.current_col:
            raise ValueError("Realised volatility can only be computed on return series.")
        return self._add_feature(
            lambda: realised_vol(self.df[self.current_col].values, window, is_sample),
            feature_name=f"rvola{window}",
            window=window,
            is_sample=is_sample
        )

    def cusum_test(self, window: int = 1000, warmup_period: int = 30) -> 'FeatureBuilder':
        """
        Apply the CUSUM test for structural breaks.

        :param window: Rolling window size for the CUSUM test.
        :param warmup_period: Warmup period for the CUSUM test.
        :return: Self for method chaining
        """
        return self._add_feature(
            lambda: cusum_test_rolling(self.df[self.current_col].values, window, warmup_period),
            feature_name=f"cusumtest{window}",
            window=window,
            warmup_period=warmup_period
        )

    def bollinger_b(self, window: int = 20, num_std: float = 2.0) -> 'FeatureBuilder':
        """
        Add Bollinger Percent B indicator to the feature set.

        :param window: Lookback window for calculations (default: 20)
        :param num_std: Number of standard deviations for bands (default: 2.0)
        :return: Self for method chaining
        """
        return self._add_feature(
            lambda: bollinger_percent_b(self.df[self.current_col].values, window, num_std),
            feature_name=f"bollingerb{window}",
            window=window,
            num_std=num_std
        )

    def vwapd(self, window: int=16, is_log: bool = True) -> 'FeatureBuilder':
        """
        Calculate the distance of the current price from the VWAP.

        :param window: Number of periods to calculate VWAP
        :param is_log: If True, compute log distance
        :return: Self for method chaining
        """
        return self._add_ohlcv_feature(
            feature_func=vwap_distance,
            required_cols=['close', 'volume'],
            feature_name=f"vwapd{window}",
            n_periods=window,
            is_log=is_log
        )

    def tr(self) -> 'FeatureBuilder':
        """
        Calculate the True Range.

        :return: Self for method chaining
        """
        return self._add_ohlcv_feature(
            feature_func=true_range,
            required_cols=['high', 'low', 'close'],
            feature_name="true_range"
        )


class FeatureKit:
    """
    Feature creation toolkit for dataframes.
    """
    def __init__(self, df: pd.DataFrame, ts_col: str = None, ohlcv_cols: dict = None):
        """
        Initialize the feature kit.

        :param df: Dataframe with the parent data
        :param ts_col: Column name containing timestamps
        :param ohlcv_cols: Dict mapping standard OHLCV names to actual column names
        """
        self.df = df  # Always work in-place
        self._ts_col = ts_col
        self._ts = self._get_timestamp()
        self._ohlcv_cols = ohlcv_cols

    def _get_timestamp(self):
        """Get timestamps as int64 numpy array"""
        if self._ts_col is None:
            logger.info("Timestamp column not specified. Using DataFrame index as timestamp.")
            return self.df.index.values.astype(np.int64)
        return self.df[self._ts_col].values.astype(np.int64)

    def feature(self, col: str) -> FeatureBuilder:
        """
        Start a feature chain from the given column.

        :param col: Column to start feature chain from
        :return: FeatureBuilder for chaining operations
        """
        return FeatureBuilder(self.df, col, self._ts, ohlcv_cols=self._ohlcv_cols)

    def ohlcv(self) -> FeatureBuilder:
        """
        Start a feature chain specifically for OHLCV data.

        :return: FeatureBuilder configured for OHLCV data
        """
        if not self._ohlcv_cols or 'close' not in self._ohlcv_cols:
            # Use default column name if not specified
            close_col = 'close'
        else:
            close_col = self._ohlcv_cols['close']

        return FeatureBuilder(self.df, close_col, self._ts, ohlcv_cols=self._ohlcv_cols)