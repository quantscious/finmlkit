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

    def __init__(self,
                 trades: pd.DataFrame,
                 interval_sec: int,
                 timestamp_unit: str = None,
                 proc_res: str = None):
        """
        Initialize the time bar builder with raw trades data and time interval.

        :param trades: DataFrame of raw trades with 'timestamp', 'price', and 'amount'.
        :param interval_sec: Time interval (in seconds) for the time bar.
        :param timestamp_unit: Optional timestamp unit; inferred if None.
        :param proc_res: Optional processing resolution.
        """
        super().__init__(trades, timestamp_unit, proc_res)
        self.interval = interval_sec

        logger.info(f"Time bar builder initialized with interval: {interval_sec} seconds.")

    def _generate_bar_opens(self) -> Tuple[NDArray[np.int64], NDArray[np.int64]]:
        """
        Generate time bar indices using the time bar indexer.
        :returns: Open timestamps and corresponding open indices in the raw trades data.
        """
        timestamps = self._raw_data['timestamp'].astype(np.int64).values
        return _time_bar_indexer(timestamps, self.interval)


class TickBarKit(BarBuilderBase):
    """
    Tick bar builder class.
    """

    def __init__(self,
                 trades: pd.DataFrame,
                 tick_count_thrs: int,
                 timestamp_unit: str = None,
                 proc_res: str = None):
        """
        Initialize the tick bar builder with raw trades data and tick count.

        :param trades: DataFrame of raw trades with 'timestamp', 'price', and 'amount'.
        :param tick_count_thrs: Tick count threshold for the tick bar.
        :param timestamp_unit: Optional timestamp unit; inferred if None.
        :param proc_res: Optional processing resolution.
        """
        super().__init__(trades, timestamp_unit, proc_res)
        self.tick_count_thrs = tick_count_thrs

        logger.info(f"Tick bar builder initialized with tick count: {tick_count_thrs}.")

    def _generate_bar_opens(self) -> Tuple[NDArray[np.int64], NDArray[np.int64]]:
        """
        Generate tick bar indices using the tick bar indexer.
        :returns: Open timestamps and corresponding open indices in the raw trades data.
        """
        timestamps = self._raw_data['timestamp'].astype(np.int64).values
        return _tick_bar_indexer(timestamps, self.tick_count_thrs)


class VolumeBarKit(BarBuilderBase):
    """
    Volume bar builder class.
    """

    def __init__(self,
                 trades: pd.DataFrame,
                 volume_thrs: float,
                 timestamp_unit: str = None,
                 proc_res: str = None):
        """
        Initialize the volume bar builder with raw trades data and volume.

        :param trades: DataFrame of raw trades with 'timestamp', 'price', and 'amount'.
        :param volume_thrs: Volume threshold for the volume bar.
        :param timestamp_unit: Optional timestamp unit; inferred if None.
        :param proc_res: Optional processing resolution.
        """
        super().__init__(trades, timestamp_unit, proc_res)
        self.volume_thrs = volume_thrs

        logger.info(f"Volume bar builder initialized with volume: {volume_thrs}.")

    def _generate_bar_opens(self) -> Tuple[NDArray[np.int64], NDArray[np.int64]]:
        """
        Generate volume bar indices using the volume bar indexer.
        :returns: Open timestamps and corresponding open indices in the raw trades data.
        """
        timestamps = self._raw_data['timestamp'].astype(np.int64).values
        volumes = self._raw_data['amount'].values
        return _volume_bar_indexer(timestamps, volumes, self.volume_thrs)


class DollarBarKit(BarBuilderBase):
    """
    Dollar bar builder class.
    """

    def __init__(self,
                 trades: pd.DataFrame,
                 dollar_thrs: float,
                 timestamp_unit: str = None,
                 proc_res: str = None):
        """
        Initialize the dollar bar builder with raw trades data and dollar amount.

        :param trades: DataFrame of raw trades with 'timestamp', 'price', and 'amount'.
        :param dollar_thrs: Dollar amount threshold for the dollar bar.
        :param timestamp_unit: Optional timestamp unit; inferred if None.
        :param proc_res: Optional processing resolution.
        """
        super().__init__(trades, timestamp_unit, proc_res)
        self.dollar_thrs = dollar_thrs

        logger.info(f"Dollar bar builder initialized with dollar amount: {dollar_thrs}.")

    def _generate_bar_opens(self) -> Tuple[NDArray[np.int64], NDArray[np.int64]]:
        """
        Generate dollar bar indices using the dollar bar indexer.
        :returns: Open timestamps and corresponding open indices in the raw trades data.
        """
        timestamps = self._raw_data['timestamp'].astype(np.int64).values
        prices = self._raw_data['price'].values
        volumes = self._raw_data['amount'].values
        return _dollar_bar_indexer(timestamps, prices, volumes, self.dollar_thrs)
