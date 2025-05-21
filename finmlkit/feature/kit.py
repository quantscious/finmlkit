from .ma import ewma, sma
from .volatility import ewmst, ewms, realised_vol, bollinger_percent_b, true_range
from .structural_break.cusum import cusum_test_rolling
from .reversion import vwap_distance
from .utils import comp_lagged_returns
from finmlkit.utils.log import get_logger
import pandas as pd
import numpy as np
from typing import Any, Callable

logger = get_logger(__name__)


class FeatureBuilder:
    """
    A chainable builder for creating features from a specific column.
    """

    def __init__(self, df: pd.DataFrame, source_col: str, timestamps: np.ndarray,
                 drop_intermediate: bool = True, ohlcv_cols: dict = None):
        """
        Initialize a feature builder for a specific column.

        :param df: Dataframe containing the data
        :param source_col: The column to build features from
        :param timestamps: Array of timestamps (int64)
        :param drop_intermediate: Whether to drop intermediate columns
        :param ohlcv_cols: Dict mapping standard OHLCV names to actual column names
                          {'open': 'Open', 'high': 'High', 'low': 'Low',
                           'close': 'Close', 'volume': 'Volume'}
        """
        self.df = df
        self.source_col = source_col
        self.current_col = source_col
        self.timestamps = timestamps
        self.drop_intermediate = drop_intermediate
        self.intermediate_cols = set()

        # Default OHLCV column mapping
        self.ohlcv_cols = {
            'open': 'open', 'high': 'high', 'low': 'low',
            'close': 'close', 'volume': 'volume'
        }
        # Override with user-specified mappings if provided
        if ohlcv_cols:
            self.ohlcv_cols.update(ohlcv_cols)

    def _add_feature(self, feature_func: Callable, out_col: str = None, **kwargs) -> 'FeatureBuilder':
        """
        Generic method to apply a feature function and manage columns.

        :param feature_func: Feature calculation function to apply
        :param out_col: Optional output column name
        :param kwargs: Parameters to pass to the feature function
        :return: Self for method chaining
        """
        # If no output column specified, generate one based on function name and params
        if not out_col:
            # Extract function name and the first parameter value for naming
            func_name = feature_func.__name__.replace("ewm", "").replace("comp_lagged_", "")
            param_value = next(iter(kwargs.values()), "")
            param_suffix = f"_{param_value}s" if isinstance(param_value, float) else f"_{param_value}"
            out_col = f"{self.current_col}_{func_name}{param_suffix}"

        # Apply the feature function
        result = feature_func(**kwargs)
        if isinstance(result, (tuple, list)):
            for i, res in enumerate(result):
                col_name = f"{out_col}_{i}"
                self.df[col_name] = res
                self.intermediate_cols.add(col_name)
        else:
            # Mark the new column as intermediate
            self.intermediate_cols.add(out_col)
            self.df[out_col] = result

        # Handle intermediate columns
        if self.drop_intermediate and self.current_col != self.source_col:
            if self.current_col in self.intermediate_cols:
                self.df.drop(columns=self.current_col, inplace=True)

        # Update current column
        self.current_col = out_col
        return self

    def _add_candle_feature(self, feature_func: Callable, required_cols: list,
                            out_col: str = None, default_name: str = None, **kwargs) -> 'FeatureBuilder':
        """
        Helper method to apply OHLCV/candle features.

        :param feature_func: Feature calculation function to apply
        :param required_cols: List of required OHLCV column types (e.g., ['close', 'volume'])
        :param out_col: Optional output column name
        :param default_name: Default name for output column if not provided
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

        # Generate output column name if not provided
        if not out_col:
            out_col = default_name

        # Extract DataFrame columns and calculate feature
        input_arrays = [self.df[col_mapping[col]].values for col in required_cols]
        self.df[out_col] = feature_func(*input_arrays, **kwargs)

        # Update state
        self.intermediate_cols.add(out_col)
        self.current_col = out_col

        return self

    def res(self) -> str:
        """Get the final column name of the feature chain."""
        return self.current_col

    def sma(self, window: int, out_col: str = None) -> 'FeatureBuilder':
        """
        Apply simple moving average (SMA) with specified window.
        :param window: Window size for the SMA calculation.
        :param out_col: Optional output column name.
        :return:
        """
        return self._add_feature(
            lambda: sma(self.df[self.current_col].values, window),
            out_col=out_col or f"{self.current_col}_sma{window}"
        )

    def ewma(self, span: int, out_col: str = None) -> 'FeatureBuilder':
        """
        Apply EWMA with specified span.
        :param span: Span for the EWMA calculation.
        :param out_col: Optional output column name.
        :return:
        """
        return self._add_feature(
            lambda: ewma(self.df[self.current_col].values, span),
            out_col=out_col or f"{self.current_col}_ewma{span}"
        )

    def ewms(self, span: int, out_col: str = None) -> 'FeatureBuilder':
        """
        Apply exponentially weighted moving standard deviation (EWMS) with specified span.
        :param span: Span for the EWMS calculation.
        :param out_col: Optional output column name.
        :return:
        """
        return self._add_feature(
            lambda: ewms(self.df[self.current_col].values, span),
            out_col=out_col or f"{self.current_col}_ewms{span}"
        )

    def ewmst(self, half_life_sec: float, out_col: str = None) -> 'FeatureBuilder':
        """
        Apply a temporal exponentially weighted moving standard deviation (EWMS) with specified half-life.
        :param half_life_sec: Half-life in seconds for the EWMS calculation.
        :param out_col: Optional output column name.
        :return:
        """
        return self._add_feature(
            lambda: ewmst(self.timestamps, self.df[self.current_col].values, half_life_sec),
            out_col=out_col or f"{self.current_col}_ewmst{half_life_sec}s"
        )

    def ret(self, return_window_sec: float, is_log: bool = False, out_col: str = None) -> 'FeatureBuilder':
        """
        Compute lagged returns over the specified time window. Works for unregular time series too.

        :param return_window_sec: Return window in seconds. Set to a small value (e.g. 1e-6) for 1 sample lag.
        :param is_log: If True, compute log returns instead of simple returns.
        :param out_col: Optional output column name.
        :return:
        """
        feat_name = "logret" if is_log else "ret"
        rws_name = f"{return_window_sec}s" if return_window_sec > 1e-6 else "1"
        return self._add_feature(
            lambda: comp_lagged_returns(self.timestamps, self.df[self.current_col].values, return_window_sec, is_log),
            out_col=out_col or f"{self.current_col}_{feat_name}{rws_name}"
        )

    def rvola(self, window: int, is_sample=False, out_col: str = None) -> 'FeatureBuilder':
        """
        Compute realised volatility over the specified period for returns.

        :param window: Number of samples to compute realised volatility over.
        :param is_sample: To calculate sample or population volatility.
        :param out_col: Optional output column name.
        :return:
        """
        # Check if the current column is a return series
        if "_ret" not in self.current_col and "_logret" not in self.current_col:
            raise ValueError("Realised volatility can only be computed on return series.")
        return self._add_feature(
            lambda: realised_vol(self.df[self.current_col].values, window, is_sample),
            out_col=out_col or f"{self.current_col}_rvola{window}"
        )

    def cusum_test(self, window: int = 1000, warmup_period: int = 30, out_col: str = None) -> 'FeatureBuilder':
        """
        Apply the CUSUM test for structural breaks.

        :param window: Rolling window size for the CUSUM test.
        :param warmup_period: Warmup period for the CUSUM test.
        :param out_col: Optional output column name.
        :return:
        """
        return self._add_feature(
            lambda: cusum_test_rolling(self.df[self.current_col].values, window, warmup_period),
            out_col=out_col or f"{self.current_col}_cusum{window}"
        )

    def bollinger_b(self, window: int = 20, num_std: float = 2.0, out_col: str = None) -> 'FeatureBuilder':
        """
        Add Bollinger Percent B indicator to the feature set.

        :param window: Lookback window for calculations (default: 20)
        :param num_std: Number of standard deviations for bands (default: 2.0)
        :param out_col: Optional output column name
        :return: Self for method chaining
        """
        return self._add_feature(
            lambda: bollinger_percent_b(self.df[self.current_col].values, window, num_std),
            out_col=out_col or f"{self.current_col}_bollb{window}"
        )

    def vwapd(self, n_periods: int=16, is_log: bool = False, out_col: str = None) -> 'FeatureBuilder':
        """
        Calculate the distance of the current price from the VWAP on a specified number of periods.

        :param n_periods: Number of periods to calculate VWAP (default: 16)
        :param is_log: If True, compute log distance.
        :param out_col: Optional output column name
        :return: Self for method chaining
        """
        return self._add_candle_feature(
            feature_func=vwap_distance,
            required_cols=['close', 'volume'],
            out_col=out_col,
            default_name=f"vwapd{n_periods}",
            n_periods=n_periods,
            is_log=is_log
        )

    def tr(self, out_col: str = None) -> 'FeatureBuilder':
        """
        Calculate the True Range.

        :param out_col: Optional output column name
        :return: Self for method chaining
        """
        return self._add_candle_feature(
            feature_func=true_range,
            required_cols=['high', 'low', 'close'],
            out_col=out_col,
            default_name="true_range"
        )


class FeatureKit:
    """
    Feature creation toolkit for dataframes.
    All feature operations are accessed through the feature() method.
    """
    def __init__(self, df: pd.DataFrame, ts_col: str = None, inplace: bool = True,
                 ohlcv_cols: dict = None):
        """
        Initialize the feature kit.

        :param df: Dataframe with the parent data
        :param ts_col: Column name containing timestamps
        :param inplace: Add feature columns to the original dataframe
        :param ohlcv_cols: Dict mapping standard OHLCV names to actual column names
        """
        self.df = df if inplace else df.copy()
        self._ts_col = ts_col
        self._ts = self._get_timestamp()
        self._ohlcv_cols = ohlcv_cols

    def _get_timestamp(self):
        """Get timestamps as int64 numpy array"""
        if self._ts_col is None:
            logger.info("Timestamp column not specified. Using DataFrame index as timestamp.")
            return self.df.index.values.astype(np.int64)
        return self.df[self._ts_col].values.astype(np.int64)

    def feature(self, col: str, drop_intermediate: bool = True) -> FeatureBuilder:
        """
        Start a feature chain from the given column.

        :param col: Column to start feature chain from
        :param drop_intermediate: Whether to drop intermediate columns
        :return: FeatureBuilder for chaining operations
        """
        return FeatureBuilder(self.df, col, self._ts, drop_intermediate)

    def ohlcv(self, drop_intermediate: bool = True) -> FeatureBuilder:
        """
        Start a feature chain specifically for OHLCV data.

        :param drop_intermediate: Whether to drop intermediate columns
        :return: FeatureBuilder configured for OHLCV data
        """
        if not self._ohlcv_cols or 'close' not in self._ohlcv_cols:
            # Use default column name if not specified
            close_col = 'close'
        else:
            close_col = self._ohlcv_cols['close']

        return FeatureBuilder(
            self.df, close_col, self._ts,
            drop_intermediate=drop_intermediate,
            ohlcv_cols=self._ohlcv_cols
        )
