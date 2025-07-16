import numpy as np
from typing import Dict, Tuple, Any
from numpy.typing import NDArray
from .base import BarBuilderBase
from .logic import _time_bar_indexer, _tick_bar_indexer, _volume_bar_indexer, _dollar_bar_indexer, _cusum_bar_indexer, _imbalance_bar_indexer, _run_bar_indexer
from finmlkit.utils.log import get_logger
from .data_model import TradesData
import pandas as pd
from finmlkit.bar.base import FootprintData, comp_volume_bar_ohlcv, comp_volume_bar_directional_features, comp_volume_bar_footprints, comp_price_tick_size
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
        self.carried_over = None

        logger.info(f"Volume bar builder initialized with volume: {volume_ths}.")

    def _comp_bar_close(self) -> Tuple[NDArray[np.int64], NDArray[np.int64]]:
        """
        Generate volume bar indices using the volume bar indexer.
        :returns: Close timestamps and corresponding close indices in the raw trades data.
        """
        timestamps = self.trades_df['timestamp'].astype(np.int64).values
        volumes = self.trades_df['amount'].values

        close_indices, carried_over = _volume_bar_indexer(volumes, self.volume_ths)
        close_indices = np.array(close_indices, dtype=np.int64)
        carried_over = np.array(carried_over, dtype=np.float64)
        self.carried_over = carried_over
        close_ts = timestamps[close_indices]

        return close_ts, close_indices

    def build_ohlcv(self) -> pd.DataFrame:
        """
        Build the bar features using the generated indices and raw trades data.
        :returns: A dataframe containing the OHLCV + VWAP features with datetime index corresponding to the bar open timestamps.
        """
        self._set_bar_close()  # Ensure bar close indices and timestamps are set

        ohlcv_tuple = comp_volume_bar_ohlcv(
            self.trades_df['price'].values,
            self.trades_df['amount'].values,
            self._close_indices,
            self.carried_over,
            self.volume_ths
        )
        self._highs, self._lows = ohlcv_tuple[1], ohlcv_tuple[2]
        logger.info("OHLCV bar calculated successfully.")

        ohlcv_df = pd.DataFrame({
            'timestamp': self.bar_close_timestamps,
            'open': ohlcv_tuple[0],
            'high': ohlcv_tuple[1],
            'low': ohlcv_tuple[2],
            'close': ohlcv_tuple[3],
            'volume': ohlcv_tuple[4],
            'trades': ohlcv_tuple[6],
            'median_trade_size': ohlcv_tuple[7],
            'vwap': ohlcv_tuple[5]
        })
        logger.info("OHLCV bar converted to DataFrame.")

        # Convert timestamps to datetime index
        ohlcv_df['timestamp'] = pd.to_datetime(ohlcv_df['timestamp'], unit='ns')
        ohlcv_df.set_index('timestamp', inplace=True)

        # if there is a self.interval attribute, set the frequency to the interval
        if hasattr(self, 'interval'):
            ohlcv_df.index.freq = pd.Timedelta(seconds=self.interval)

        return ohlcv_df

    def build_directional_features(self) -> pd.DataFrame:
        """
        Build the directional features using the generated indices and raw trades data.
        :returns: A dataframe containing the directional features:
            ticks_buy, ticks_sell, volume_buy, volume_sell, dollars_buy, dollars_sell, max_spread,
            cum_volumes_min, cum_volumes_max, cum_dollars_min, cum_dollars_max.
        """
        self._set_bar_close()

        directional_tuple = comp_volume_bar_directional_features(
            self.trades_df['price'].values,
            self.trades_df['amount'].values,
            self._close_indices,
            self.trades_df['side'].values.astype(np.int8),
            self.carried_over,
            self.volume_ths
        )
        logger.info("Directional features calculated successfully.")

        directional_df = pd.DataFrame({
            'timestamp': self.bar_close_timestamps,     # Close bar timestamps convention!!
            'ticks_buy': directional_tuple[0],
            'ticks_sell': directional_tuple[1],
            'volume_buy': directional_tuple[2],
            'volume_sell': directional_tuple[3],
            'dollars_buy': directional_tuple[4],
            'dollars_sell': directional_tuple[5],
            'mean_spread': directional_tuple[6],
            'max_spread': directional_tuple[7],
            'cum_ticks_min': directional_tuple[8],
            'cum_ticks_max': directional_tuple[9],
            'cum_volume_min': directional_tuple[10],
            'cum_volume_max': directional_tuple[11],
            'cum_dollars_min': directional_tuple[12],
            'cum_dollars_max': directional_tuple[13]
        })
        logger.info("Directional features converted to DataFrame.")

        # Convert timestamps to datetime index
        directional_df['timestamp'] = pd.to_datetime(directional_df['timestamp'], unit='ns')
        directional_df.set_index('timestamp', inplace=True)

        return directional_df

    def build_footprints(self, price_tick_size=None, imbalance_factor=3.0) -> FootprintData:
        """
        Build the footprint data using the generated indices and raw trades data.
        :param price_tick_size: Optional tick size; inferred if None.
        :param imbalance_factor: Multiplier for detecting imbalances. Default is 3.0.
        :returns: A FootprintData object containing the footprint data.
        """

        self._set_bar_close()  # Ensure bar close indices and timestamps are set
        if self._highs is None or self._lows is None:
            # We need the bar highs and lows for the footprint calculation
            self.build_ohlcv()

        if price_tick_size is None:
            # Infer price tick size
            price_tick_size = comp_price_tick_size(self.trades_df['price'].values)
        logger.info(f"Price tick size is set to: {price_tick_size}")

        # Compute the footprint data
        footprint_data = comp_volume_bar_footprints(
            self.trades_df['price'].values,
            self.trades_df['amount'].values,
            self._close_indices,
            self.trades_df['side'].values.astype(np.int8),
            self.carried_over,
            self.volume_ths,
            price_tick_size,
            self._lows,
            self._highs,
            imbalance_factor
        )
        logger.info("Footprint data calculated successfully.")

        # Create a FootprintData object with all metrics
        footprint = FootprintData(
            bar_timestamps= self.bar_close_timestamps,
            price_levels=footprint_data[0],
            price_tick=price_tick_size,
            buy_volumes=footprint_data[1],
            sell_volumes=footprint_data[2],
            buy_ticks=footprint_data[3],
            sell_ticks=footprint_data[4],
            buy_imbalances=footprint_data[5],
            sell_imbalances=footprint_data[6],
            buy_imbalances_sum=footprint_data[7],
            sell_imbalances_sum=footprint_data[8],
            cot_price_levels=footprint_data[9],
            imb_max_run_signed=footprint_data[10],
            vp_skew=footprint_data[11],
            vp_gini=footprint_data[12]
        )
        footprint.cast_to_numba_list()
        logger.info("Footprint data converted to FootprintData object.")

        return footprint


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