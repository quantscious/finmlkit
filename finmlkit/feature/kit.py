from .ma import ewma
from .volatility import ewmst, ewms
from .utils import compute_lagged_returns
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
            func_name = feature_func.__name__.replace("ewm", "").replace("compute_lagged_", "")
            param_value = next(iter(kwargs.values()), "")
            param_suffix = f"_{param_value}s" if isinstance(param_value, float) else f"_{param_value}"
            out_col = f"{self.current_col}_{func_name}{param_suffix}"

        # Apply the feature function
        self.df[out_col] = feature_func(**kwargs)

        # Handle intermediate columns
        if self.drop_intermediate and self.current_col != self.source_col:
            if self.current_col in self.intermediate_cols:
                self.df.drop(columns=self.current_col, inplace=True)

        # Mark the new column as intermediate
        self.intermediate_cols.add(out_col)
        # Update current column
        self.current_col = out_col
        return self

    def ewma(self, span: int, out_col: str = None) -> 'FeatureBuilder':
        """Apply EWMA with specified span."""
        return self._add_feature(
            lambda: ewma(self.df[self.current_col].values, span),
            out_col=out_col or f"{self.current_col}_ewma_{span}"
        )

    def ewms(self, span: int, out_col: str = None) -> 'FeatureBuilder':
        """Apply EWMS with specified span."""
        return self._add_feature(
            lambda: ewms(self.df[self.current_col].values, span),
            out_col=out_col or f"{self.current_col}_ewms_{span}"
        )

    def ewmst(self, half_life_sec: float, out_col: str = None) -> 'FeatureBuilder':
        """Apply time-decay EWMS."""
        return self._add_feature(
            lambda: ewmst(self.timestamps, self.df[self.current_col].values, half_life_sec),
            out_col=out_col or f"{self.current_col}_ewmst_{half_life_sec}s"
        )

    def ret(self, return_window_sec: float, out_col: str = None) -> 'FeatureBuilder':
        """Compute lagged returns over the specified window."""
        return self._add_feature(
            lambda: compute_lagged_returns(self.timestamps, self.df[self.current_col].values, return_window_sec),
            out_col=out_col or f"{self.current_col}_ret_{return_window_sec}s"
        )

    def res(self) -> str:
        """Get the final column name of the feature chain."""
        return self.current_col


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



