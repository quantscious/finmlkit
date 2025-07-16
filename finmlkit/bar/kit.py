import numpy as np
from typing import Dict, Tuple, Any
from numpy.typing import NDArray
from .base import BarBuilderBase
from .logic import _time_bar_indexer, _tick_bar_indexer, _volume_bar_indexer, _dollar_bar_indexer, _cusum_bar_indexer, _imbalance_bar_indexer, _run_bar_indexer
from finmlkit.utils.log import get_logger
from .data_model import TradesData
import pandas as pd
logger = get_logger(__name__)


class TimeBarKit(BarBuilderBase):
    """
    Time bar builder class.
    """

    def __init__(self,trades: TradesData, period: pd.Timedelta):
        """
        Initialize the time bar builder with raw trades data and time interval.

        :param trades: DataFrame of raw trades with 'timestamp', 'price', and 'amount'.
        :param period: The time interval of a bar.
        """
        super().__init__(trades)
        self.interval = period.total_seconds()

        logger.info(f"Time bar builder initialized with interval: {self.interval} seconds.")

    def _comp_bar_close(self) -> Tuple[NDArray[np.int64], NDArray[np.int64]]:
        """
        Generate time bar indices using the time bar indexer.
        :returns: Close timestamps and corresponding Close indices in the raw trades data.
        """
        timestamps = self.trades_df['timestamp'].astype(np.int64).values
        return _time_bar_indexer(timestamps, self.interval)


class TickBarKit(BarBuilderBase):
    """
    Tick bar builder class.
    """

    def __init__(self,
                 trades: TradesData,
                 tick_count_thrs: int):
        """
        Initialize the tick bar builder with raw trades data and tick count.

        :param trades: DataFrame of raw trades with 'timestamp', 'price', and 'amount'.
        :param tick_count_thrs: Tick count threshold for the tick bar.
        """
        super().__init__(trades)
        self.tick_count_thrs = tick_count_thrs

        logger.info(f"Tick bar builder initialized with tick count: {tick_count_thrs}.")

    def _comp_bar_close(self) -> Tuple[NDArray[np.int64], NDArray[np.int64]]:
        """
        Generate tick bar indices using the tick bar indexer.
        :returns: Close timestamps and corresponding close indices in the raw trades data.
        """
        timestamps = self.trades_df['timestamp'].astype(np.int64).values
        close_indices = _tick_bar_indexer(timestamps, self.tick_count_thrs)
        close_indices = np.array(close_indices, dtype=np.int64)
        close_ts = timestamps[close_indices]

        return close_ts, close_indices


class VolumeBarKit(BarBuilderBase):
    """
    Volume bar builder class.
    """

    def __init__(self,
                 trades: TradesData,
                 volume_ths: float):
        """
        Initialize the volume bar builder with raw trades data and volume.

        :param trades: DataFrame of raw trades with 'timestamp', 'price', and 'amount'.
        :param volume_ths: Volume Bucket threshold for the volume bar.
        """
        super().__init__(trades)
        self.volume_ths = volume_ths

        logger.info(f"Volume bar builder initialized with volume: {volume_ths}.")

    def _comp_bar_close(self) -> Tuple[NDArray[np.int64], NDArray[np.int64]]:
        """
        Generate volume bar indices using the volume bar indexer.
        :returns: Close timestamps and corresponding close indices in the raw trades data.
        """
        timestamps = self.trades_df['timestamp'].astype(np.int64).values
        volumes = self.trades_df['amount'].values

        close_indices = _volume_bar_indexer(volumes, self.volume_ths)
        close_indices = np.array(close_indices, dtype=np.int64)
        close_ts = timestamps[close_indices]

        return close_ts, close_indices



class DollarBarKit(BarBuilderBase):
    """
    Dollar bar builder class.
    """

    def __init__(self,
                 trades: TradesData,
                 dollar_thrs: float):
        """
        Initialize the dollar bar builder with raw trades data and dollar amount.

        :param trades: DataFrame of raw trades with 'timestamp', 'price', and 'amount'.
        :param dollar_thrs: Dollar amount threshold for the dollar bar.
        """
        super().__init__(trades)
        self.dollar_thrs = dollar_thrs

        logger.info(f"Dollar bar builder initialized with dollar amount: {dollar_thrs}.")

    def _comp_bar_close(self) -> Tuple[NDArray[np.int64], NDArray[np.int64]]:
        """
        Generate dollar bar indices using the dollar bar indexer.
        :returns: Close timestamps and corresponding close indices in the raw trades data.
        """
        timestamps = self.trades_df['timestamp'].astype(np.int64).values
        prices = self.trades_df['price'].values
        volumes = self.trades_df['amount'].values

        close_indices = _dollar_bar_indexer(prices, volumes, self.dollar_thrs)
        close_indices = np.array(close_indices, dtype=np.int64)
        close_ts = timestamps[close_indices]

        return close_ts, close_indices


class CUSUMBarKit(BarBuilderBase):
    def __init__(self,
                 trades: TradesData,
                 sigma: NDArray[np.float64],
                 sigma_floor: float = 5e-4,
                 sigma_mult: float = 2.
                 ):
        """
        Initialize the CUSUM bar builder with raw trades data and threshold.

        :param trades: DataFrame of raw trades with 'timestamp', 'price', and 'amount'.
        :param sigma: Standard deviation vector of the price series or a constant value for all ticks.
        :param sigma_floor: Minimum value for sigma to avoid small events.
        :param sigma_mult: the sigma multiplier for adaptive threshold (lambda_th = lambda_mult * sigma).
        """
        super().__init__(trades)
        self.lambda_mult = sigma_mult
        self._sigma = sigma
        self.sigma_floor = sigma_floor

        logger.info(f"CUSUM Bar builder initialized with: sigma multiplier={sigma_mult}.")

    def _comp_bar_close(self) -> Tuple[NDArray[np.int64], NDArray[np.int64]]:
        """
        Generate CUSUM bar indices using the CUSUM bar indexer.
        :returns: Open timestamps and corresponding open indices in the raw trades data.
        """
        timestamps = self.trades_df['timestamp'].astype(np.int64).values
        prices = self.trades_df['price'].values

        close_indices = _cusum_bar_indexer(timestamps, prices, self._sigma, self.sigma_floor, self.lambda_mult)
        close_indices = np.array(close_indices, dtype=np.int64)
        close_ts = timestamps[close_indices]

        return close_ts, close_indices

    def get_sigma(self) -> NDArray[np.float64]:
        """
        The sigma threshold used for the CUSUM at close indices.
        :return: sigma vector
        """
        return self._sigma[self.bar_close_indices]