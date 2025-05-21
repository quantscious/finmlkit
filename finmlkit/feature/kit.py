from .ma import ewma
from .volatility import ewmst, ewms, realised_vol
from .structural_break.cusum import cusum_test_rolling
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
    def __init__(self, df: pd.DataFrame, source_col: str, timestamps: np.ndarray, drop_intermediate: bool = True):
        """
        Initialize a feature builder for a specific column.

        :param df: Dataframe containing the data
        :param source_col: The column to build features from
        :param timestamps: Array of timestamps (int64)
        :param drop_intermediate: Whether to drop intermediate columns
        """
        self.df = df
        self.source_col = source_col  # Original source column (never dropped)
        self.current_col = source_col  # Last output column
        self.timestamps = timestamps
        self.drop_intermediate = drop_intermediate
        self.intermediate_cols = set()  # Track intermediate columns

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

    def res(self) -> str:
        """Get the final column name of the feature chain."""
        return self.current_col

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
        rws_name = f"{return_window_sec}s" if return_window_sec > 1e-6 else "1step"
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
            out_col=out_col or f"{self.current_col}_rvola{window}s"
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


class FeatureKit:
    """
    Feature creation toolkit for dataframes.
    All feature operations are accessed through the feature() method.
    """
    def __init__(self, df: pd.DataFrame, ts_col: str = None, inplace: bool = True):
        """
        Initialize the feature kit.

        :param df: Dataframe with the parent data
        :param ts_col: Column name containing timestamps
        :param inplace: Add feature columns to the original dataframe
        """
        self.df = df if inplace else df.copy()
        self._ts_col = ts_col
        self._ts = self._get_timestamp()

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



