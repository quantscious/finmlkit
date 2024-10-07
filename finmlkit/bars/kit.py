import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Any
from numpy.typing import NDArray
from .build import comp_bar_ohlcv, comp_bar_directional_features, comp_bar_footprints
from .logic import time_bar_indexer, tick_bar_indexer, volume_bar_indexer, dollar_bar_indexer, imbalance_bar_indexer, run_bar_indexer


class BarBuilderBase(ABC):
    """
    This class provides a template for generating bars from raw trades data.
    """

    def __init__(self, trades: pd.DataFrame):
        """
        Initialize the bar builder with raw trades data.

        Parameters
        ----------
        trades : pd.DataFrame
            A dataframe containing raw trades data with columns 'timestamp', 'price', and 'amount'.
        """
        self._raw_data = trades
        self._open_indices = None

    @abstractmethod
    def generate_bar_indices(self) -> Tuple[NDArray[np.int64], NDArray[np.int64]]:
        """
        Abstract method to generate bar indices using the appropriate bar indexer.
        Returns
        -------
        tuple(np.array(np.int64), np.array(np.int64))
            Open timestamps and corresponding open indices in the raw trades data.
        """
        pass

    def build_bar_features(self) -> Dict[str, NDArray]:
        """
        Build the bar features using the generated indices and raw trades data.
        Returns
        -------
        dict
            A dictionary containing the computed bar features.
        """
        open_timestamps, open_indices = self.generate_bar_indices()
        prices = self._raw_data['prices']
        volumes = self._raw_data['volumes']

        # Construct OHLCV and other features using the building functions
        ohlcv = comp_bar_ohlcv(prices, volumes, open_indices)
        directional_features = comp_bar_directional_features(prices, volumes, open_indices)
        footprints = comp_bar_footprints(prices, volumes, open_indices)

        return {
            "ohlcv": ohlcv,
            "directional_features": directional_features,
            "footprints": footprints
        }


class TimeBarBuilder(BarBuilderBase):
    """
    Time bar builder class.
    """

    def __init__(self, trades: pd.DataFrame, interval: int):
        """
        Initialize the time bar builder with raw trades data and time interval.

        Parameters
        ----------
        trades : pd.DataFrame
            A dataframe containing raw trades data with columns 'timestamp', 'price', and 'amount'.
        interval : int
            The time interval in seconds for the time bars.
        """
        super().__init__(trades)
        self.interval = interval

    def generate_bar_indices(self) -> Tuple[NDArray[np.int64], NDArray[np.int64]]:
        """
        Generate time bar indices using the time bar indexer.
        Returns
        -------
        tuple(np.array(np.int64), np.array(np.int64))
            Open timestamps and corresponding open indices in the raw trades data.
        """
        timestamps = self._raw_data['timestamp'].astype(np.int64).values
        return time_bar_indexer(timestamps, self.interval)