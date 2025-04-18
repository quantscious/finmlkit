import numpy as np
import pandas as pd
from typing import Dict, Tuple, Any
from numpy.typing import NDArray
from .base import BarBuilderBase
from .logic import _time_bar_indexer, _tick_bar_indexer, _volume_bar_indexer, _dollar_bar_indexer, _imbalance_bar_indexer, _run_bar_indexer
from finmlkit.utils.log import logger


class TimeBarKit(BarBuilderBase):
    """
    Time bar builder class.
    """

    def __init__(self, trades: pd.DataFrame, interval_sec: int):
        """
        Initialize the time bar builder with raw trades data and time interval.

        Parameters
        ----------
        trades : pd.DataFrame
            A dataframe containing raw trades data with columns 'timestamp', 'price', and 'amount'.
        interval : int
            The time interval in seconds for the time bar.
        """
        super().__init__(trades)
        self.interval = interval_sec

        logger.info(f"Time bar builder initialized with interval: {interval_sec} seconds.")

    def _generate_bar_opens(self) -> Tuple[NDArray[np.int64], NDArray[np.int64]]:
        """
        Generate time bar indices using the time bar indexer.
        Returns
        -------
        tuple(np.array(np.int64), np.array(np.int64))
            Open timestamps and corresponding open indices in the raw trades data.
        """
        timestamps = self._raw_data['timestamp'].astype(np.int64).values
        return _time_bar_indexer(timestamps, self.interval)


class TickBarKit(BarBuilderBase):
    """
    Tick bar builder class.
    """

    def __init__(self, trades: pd.DataFrame, tick_count_thrs: int):
        """
        Initialize the tick bar builder with raw trades data and tick count.

        Parameters
        ----------
        trades : pd.DataFrame
            A dataframe containing raw trades data with columns 'timestamp', 'price', and 'amount'.
        tick_count : int
            The tick count for the tick bar.
        """
        super().__init__(trades)
        self.tick_count_thrs = tick_count_thrs

        logger.info(f"Tick bar builder initialized with tick count: {tick_count_thrs}.")

    def _generate_bar_opens(self) -> Tuple[NDArray[np.int64], NDArray[np.int64]]:
        """
        Generate tick bar indices using the tick bar indexer.
        Returns
        -------
        tuple(np.array(np.int64), np.array(np.int64))
            Open timestamps and corresponding open indices in the raw trades data.
        """
        timestamps = self._raw_data['timestamp'].astype(np.int64).values
        return _tick_bar_indexer(timestamps, self.tick_count_thrs)


class VolumeBarKit(BarBuilderBase):
    """
    Volume bar builder class.
    """

    def __init__(self, trades: pd.DataFrame, volume_thrs: float):
        """
        Initialize the volume bar builder with raw trades data and volume.

        Parameters
        ----------
        trades : pd.DataFrame
            A dataframe containing raw trades data with columns 'timestamp', 'price', and 'amount'.
        volume : float
            The volume for the volume bar.
        """
        super().__init__(trades)
        self.volume_thrs = volume_thrs

        logger.info(f"Volume bar builder initialized with volume: {volume_thrs}.")

    def _generate_bar_opens(self) -> Tuple[NDArray[np.int64], NDArray[np.int64]]:
        """
        Generate volume bar indices using the volume bar indexer.
        Returns
        -------
        tuple(np.array(np.int64), np.array(np.int64))
            Open timestamps and corresponding open indices in the raw trades data.
        """
        timestamps = self._raw_data['timestamp'].astype(np.int64).values
        volumes = self._raw_data['amount'].values
        return _volume_bar_indexer(timestamps, volumes, self.volume_thrs)


class DollarBarKit(BarBuilderBase):
    """
    Dollar bar builder class.
    """

    def __init__(self, trades: pd.DataFrame, dollar_thrs: float):
        """
        Initialize the dollar bar builder with raw trades data and dollar amount.

        Parameters
        ----------
        trades : pd.DataFrame
            A dataframe containing raw trades data with columns 'timestamp', 'price', and 'amount'.
        dollar : float
            The dollar amount for the dollar bar.
        """
        super().__init__(trades)
        self.dollar_thrs = dollar_thrs

        logger.info(f"Dollar bar builder initialized with dollar amount: {dollar_thrs}.")

    def _generate_bar_opens(self) -> Tuple[NDArray[np.int64], NDArray[np.int64]]:
        """
        Generate dollar bar indices using the dollar bar indexer.
        Returns
        -------
        tuple(np.array(np.int64), np.array(np.int64))
            Open timestamps and corresponding open indices in the raw trades data.
        """
        timestamps = self._raw_data['timestamp'].astype(np.int64).values
        prices = self._raw_data['price'].values
        volumes = self._raw_data['amount'].values
        return _dollar_bar_indexer(timestamps, prices, volumes, self.dollar_thrs)
