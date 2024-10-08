import numpy as np
import pandas as pd
from typing import Dict, Tuple, Any
from numpy.typing import NDArray
from .build import BarBuilderBase
from .logic import _time_bar_indexer, _tick_bar_indexer, _volume_bar_indexer, _dollar_bar_indexer, _imbalance_bar_indexer, _run_bar_indexer
from finmlkit.utils.log import logger


class TimeBarKit(BarBuilderBase):
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

        logger.info(f"Time bar builder initialized with interval: {interval} seconds.")

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