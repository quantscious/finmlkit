"""
This module contains the functions to build candlestick bar and other intra-bar features
(i.e. directional features, footprints) from raw trades data using the indexer functions outputs
defined in the logic module.
"""
import numpy as np
from numba import njit
from numba import prange
from typing import Tuple, Optional
import pandas as pd
from numba.typed import List as NumbaList
from abc import ABC, abstractmethod
from numpy.typing import NDArray
import io

from .data_model import FootprintData
from .utils import comp_price_tick_size
from .data_model import TradesData

from finmlkit.utils.log import get_logger
logger = get_logger(__name__)


class BarBuilderBase(ABC):
    r"""Abstract base class for building various types of bars (e.g., time, tick, volume, or information based bars) from raw trades data.
    This class serves as a template for subclasses that implement specific bar sampling strategies, enabling the transformation
    of high-frequency trade data into structured bar features suitable for financial analysis and machine learning.

    In financial machine learning, raw trade data (ticks) is often aggregated into bars to reduce noise, capture market dynamics,
    and create features for modeling. This builder computes standard OHLCV (Open, High, Low, Close, Volume) bars, directional features
    (e.g., buy/sell volumes), trade size metrics, and footprint data (order flow imbalances at price levels). It is inspired by
    techniques from Marcos López de Prado's work on sampling methods to address issues like uneven information arrival rates
    in high-frequency trading data.

    Subclasses must implement the abstract method :meth:`_comp_bar_close` to define how bar close timestamps and indices are determined
    (e.g., based on time intervals, tick counts, or volume thresholds). The builder uses these indices to aggregate trades efficiently
    via Numba and Pandas, ensuring performance for large datasets.

    Key functionalities include:

    - :meth:`build_ohlcv`: Computes OHLCV, VWAP (Volume-Weighted Average Price), trade count, and median trade size.
    - :meth:`build_directional_features`: Calculates buy/sell splits for ticks, volume, dollar value, spreads, and cumulative metrics,
      revealing order flow directionality and market pressure.
    - :meth:`build_trade_size_features`: Analyzes relative trade sizes, 95th percentile sizes, block trade percentages, and Gini coefficients
        for trade size distribution, useful for detecting large orders or market concentration.
    - :meth:`build_footprints`: Generates detailed footprint data, discretizing price levels to compute volumes, ticks, imbalances,
      and metrics like volume profile skew and Gini, aiding in order flow and volume profile analysis.


    Args:
        trades (TradesData): Object containing raw trades DataFrame with columns 'timestamp', 'price', and 'amount'.
            TradesData ensures the data is preprocessed and ready for bar construction.

    Raises:
        ValueError: If required columns are missing from trades data or if data is not properly formatted.

    See Also:
        :class:`finmlkit.bar.kit.TimeBarKit`: A concrete subclass for fixed-time interval bars.

        :class:`finmlkit.bar.kit.TickBarKit`: For bars based on tick counts.

        :class:`finmlkit.bar.kit.VolumeBarKit`: For volume-threshold bars.
    """

    def __init__(self, trades: TradesData):
        """
        Initialize the bar builder with raw trades data.

        :param trades: TradesData object containing raw trades DataFrame with columns 'timestamp', 'price', and 'amount'.
        """
        self.trades_df = trades.data

        self._close_ts:      Optional[NDArray[np.int64]] = None
        self._close_indices: Optional[NDArray[np.int64]] = None
        self._highs:        Optional[NDArray[np.float64]] = None
        self._lows:         Optional[NDArray[np.float64]] = None

    def __str__(self) -> str:
        members = "\n".join(f"{k}: {v}" for k, v in self.__dict__.items())
        buf = io.StringIO()
        try:
            self.trades_df.info(buf=buf)
            info = buf.getvalue()
        except Exception:
            info = "<unavailable>"
        return (
            f"Class: {self.__class__.__name__} with members:\n"
            f"{members}\n"
            f"Raw trades data:\n{info}"
        )

    @abstractmethod
    def _comp_bar_close(self) -> Tuple[NDArray[np.int64], NDArray[np.int64]]:
        """
        Abstract method to generate bar close timestamps and indices.

        :returns: Tuple of close timestamps and their corresponding indices.
        """
        pass

    def _set_bar_close(self):
        """
        Calculate and sets the close timestamps and indices if not already calculated.
        """
        if self._close_ts is None and self._close_indices is None:
            logger.info("Calculating bar close tick indices and timestamps...")
            self._close_ts, self._close_indices = self._comp_bar_close()

    @property
    def bar_close_indices(self) -> Optional[NDArray[np.int64]]:
        """
        Return the bar close indices in the raw trades data.

        :return: The **bar close** indices regarding the raw trades data as a numpy array of int64.
        """
        if self._close_indices is None:
            self._set_bar_close()
        return self._close_indices[1:]  # Exclude the first timestamp as it is the bar open timestamp

    @property
    def bar_close_timestamps(self) -> Optional[NDArray[np.int64]]:
        """
        Return the bar close timestamps in the raw trades data.

        :return: The **bar close** ns timestamps as a numpy array of int64.
        """
        if self._close_ts is None:
            self._set_bar_close()
        return self._close_ts[1:]  # Exclude the first timestamp as it is the bar open timestamp


    def build_ohlcv(self) -> pd.DataFrame:
        """
        Build the bar features using the generated indices and raw trades data.

        :returns: A dataframe containing the OHLCV + VWAP features with datetime index corresponding to the bar open timestamps.
        """
        self._set_bar_close()  # Ensure bar close indices and timestamps are set

        ohlcv_tuple = comp_bar_ohlcv(
            self.trades_df['price'].values,
            self.trades_df['amount'].values,
            self._close_indices
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

        directional_tuple = comp_bar_directional_features(
            self.trades_df['price'].values,
            self.trades_df['amount'].values,
            self._close_indices,
            self.trades_df['side'].values.astype(np.int8),
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

    def build_trade_size_features(self, theta: Optional[NDArray[np.float64]], theta_mult: float = 5.0) -> pd.DataFrame:
        """
        Build the trade size features using the generated indices and raw trades data.
        :param theta: Optional typical trade size (e.g., 30 day rolling median trade size).
        :param theta_mult: Multiplier for theta to define the block size threshold. Default is 5.0.
        :returns: A dataframe containing the trade size features:
            mean_size_rel, size_95_rel, pct_block, size_gini.
        """
        self._set_bar_close()  # Ensure bar close indices and timestamps are set

        trade_size_tuple = comp_bar_trade_size_features(
            self.trades_df['amount'].values,
            theta,
            self._close_indices,
            theta_mult
        )
        logger.info("Trade size features calculated successfully.")

        trade_size_df = pd.DataFrame({
            'timestamp': self.bar_close_timestamps,
            'mean_size_rel': trade_size_tuple[0],
            'size_95_rel': trade_size_tuple[1],
            'pct_block': trade_size_tuple[2],
            'size_gini': trade_size_tuple[3]
        })
        logger.info("Trade size features converted to DataFrame.")

        # Convert timestamps to datetime index
        trade_size_df['timestamp'] = pd.to_datetime(trade_size_df['timestamp'], unit='ns')
        trade_size_df.set_index('timestamp', inplace=True)

        return trade_size_df

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
        footprint_data = comp_bar_footprints(
            self.trades_df['price'].values,
            self.trades_df['amount'].values,
            self._close_indices,
            self.trades_df['side'].values.astype(np.int8),
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


# --------------------------------------------------------------------------------------------
# CORE FUNCTIONS
# --------------------------------------------------------------------------------------------
@njit(nogil=True, parallel=True)
def comp_bar_ohlcv(
        prices: NDArray[np.float64],
        volumes: NDArray[np.float64],
        bar_close_indices: NDArray[np.int64],
) -> tuple[
    NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float32], NDArray[
        np.float64], NDArray[np.int64], NDArray[np.float64]]:
    """
    Build the candlestick bar from raw trades data based on bar close indices.

    :param prices: Trade prices.
    :param volumes: Trade volumes.
    :param bar_close_indices: Indices marking the end of each bar.
    :returns: Tuple containing:

        - open: Opening price of each bar.
        - high: Highest price of each bar.
        - low: Lowest price of each bar.
        - close: Closing price of each bar.
        - volume: Total traded volume in each bar.
        - vwap: Volume-weighted average price of each bar.
        - bar_trades: Number of trades in each bar.
        - bar_median_trade_size: Median trade size in each bar.
    """
    # Check the input arrays match in length
    if len(prices) != len(volumes):
        raise ValueError("Prices and volumes arrays must have the same length.")
    if len(bar_close_indices) < 2:
        raise ValueError("Bar close indices must contain at least two elements.")

    n_bars = len(bar_close_indices) - 1
    bar_high = np.zeros(n_bars, dtype=np.float64)
    bar_low = np.zeros(n_bars, dtype=np.float64)
    bar_open = np.zeros(n_bars, dtype=np.float64)
    bar_close = np.zeros(n_bars, dtype=np.float64)
    bar_volume = np.zeros(n_bars, dtype=np.float32)
    bar_trades = np.zeros(n_bars, dtype=np.int64)
    bar_median_trade_size = np.zeros(n_bars, dtype=np.float64)
    bar_vwap = np.zeros(n_bars, dtype=np.float64)

    for i in prange(n_bars):
        start = bar_close_indices[i]
        end = bar_close_indices[i + 1]

        # Handle empty bar
        if start == end:
            bar_open[i] = prices[end]
            bar_close[i] = prices[end]
            bar_high[i] = prices[end]
            bar_low[i] = prices[end]
            bar_volume[i] = 0.0
            bar_vwap[i] = 0.0
            bar_trades[i] = 0
            bar_median_trade_size[i] = 0.0
            continue

        # Start from the next trade (start=previous bar close)
        start += 1
        # Initialize variables for this bar
        high_price = prices[start]
        low_price = prices[start]
        total_volume = 0.0
        total_dollar = 0.0

        # Count number of trades and collect trade sizes for median calculation
        trade_count = end - start + 1
        trade_sizes = np.zeros(trade_count, dtype=np.float64)
        trade_idx = 0

        # Iterate over trades in the current bar, inclusive for the last trade (bar close)
        for j in range(start, end + 1):
            price = prices[j]
            volume = volumes[j]

            # Store trade size for median calculation
            trade_sizes[trade_idx] = volume
            trade_idx += 1

            if price > high_price:
                high_price = price
            if price < low_price:
                low_price = price

            total_volume += volume
            total_dollar += price * volume

        bar_open[i] = prices[start]  # First trade price in the bar exclusive
        bar_close[i] = prices[end]  # Last trade price in the bar inclusive
        bar_high[i] = high_price
        bar_low[i] = low_price
        bar_volume[i] = total_volume
        bar_vwap[i] = total_dollar / total_volume if total_volume > 0 else 0.0
        bar_trades[i] = trade_count

        # Calculate median trade size
        if trade_count > 0:
            bar_median_trade_size[i] = np.median(trade_sizes)
        else:
            bar_median_trade_size[i] = 0.0

    return bar_open, bar_high, bar_low, bar_close, bar_volume, bar_vwap, bar_trades, bar_median_trade_size

@njit(nogil=True, parallel=True)
def comp_bar_directional_features(
    prices: NDArray[np.float64],
    volumes: NDArray[np.float64],
    bar_close_indices: NDArray[np.int64],
    trade_sides: NDArray[np.int8]
) -> tuple[
    NDArray[np.int64], NDArray[np.int64],
    NDArray[np.float32], NDArray[np.float32],
    NDArray[np.float32], NDArray[np.float32],
    NDArray[np.float32], NDArray[np.float32],
    NDArray[np.int64], NDArray[np.int64],
    NDArray[np.float32], NDArray[np.float32],
    NDArray[np.float32], NDArray[np.float32]
]:
    """
    Compute directional bar features such as tick counts, volumes, dollars, spreads, and cumulative flows.

    :param prices: Trade prices.
    :param volumes: Trade volumes.
    :param bar_close_indices: Indices marking the end of each bar.
    :param trade_sides: Trade direction (1 for market buy, -1 for market sell).
    :returns: Tuple containing:

        - ticks_buy: Number of buy trades per bar.
        - ticks_sell: Number of sell trades per bar.
        - volume_buy: Volume of buy trades per bar.
        - volume_sell: Volume of sell trades per bar.
        - dollars_buy: Dollar value of buy trades per bar.
        - dollars_sell: Dollar value of sell trades per bar.
        - mean_spread: Mean bid/ask spread within each bar.
        - max_spread: Maximum spread within each bar.
        - cum_ticks_min: Minimum cumulative tick imbalance.
        - cum_ticks_max: Maximum cumulative tick imbalance.
        - cum_volumes_min: Minimum cumulative volume imbalance.
        - cum_volumes_max: Maximum cumulative volume imbalance.
        - cum_dollars_min: Minimum cumulative dollar imbalance.
        - cum_dollars_max: Maximum cumulative dollar imbalance.
    """
    n_bars = len(bar_close_indices) - 1
    ticks_buy = np.zeros(n_bars, dtype=np.int64)
    ticks_sell = np.zeros(n_bars, dtype=np.int64)
    volume_buy = np.zeros(n_bars, dtype=np.float32)
    volume_sell = np.zeros(n_bars, dtype=np.float32)
    dollars_buy = np.zeros(n_bars, dtype=np.float32)
    dollars_sell = np.zeros(n_bars, dtype=np.float32)
    max_spread = np.zeros(n_bars, dtype=np.float32)
    mean_spread = np.zeros(n_bars, dtype=np.float32)

    # Initialize cumulative min and max arrays with appropriate values
    cum_ticks_min = np.full(n_bars, 1e9, dtype=np.int64)            # inf (large value)
    cum_ticks_max = np.full(n_bars, -1e9, dtype=np.int64)                   # -inf (small value)
    cum_volumes_min = np.full(n_bars, 1e9, dtype=np.float32)
    cum_volumes_max = np.full(n_bars, -1e9, dtype=np.float32)
    cum_dollars_min = np.full(n_bars, 1e9, dtype=np.float32)
    cum_dollars_max = np.full(n_bars, -1e9, dtype=np.float32)

    # Compute the bar directional features
    for i in prange(n_bars):
        start = bar_close_indices[i] + 1  # Start from the next trade (start=previous bar close)
        end = bar_close_indices[i + 1]

        current_tics_buy = 0
        current_tics_sell = 0
        current_volume_buy = 0.0
        current_volume_sell = 0.0
        current_dollars_buy = 0.0
        current_dollars_sell = 0.0
        # Cumulative values
        current_cum_ticks = 0
        current_cum_volumes = 0.0
        current_cum_dollars = 0.0
        current_max_spread = 0.0
        current_cum_spread = 0.0

        # Initialize previous tick sign for spread calculation
        if end > start:
            prev_tick_sign = trade_sides[start - 1]  # Previous trade side at the start of the bar
        else:
            prev_tick_sign = 0  # Default value if no trades in bar

        # Iterate over trades in the current bar (start exclusive, end inclusive)
        for j in range(start, end + 1):
            current_tick_sign = trade_sides[j]

            # Calculate the spread between buy and sell prices
            if current_tick_sign != prev_tick_sign:
                spread = abs(prices[j] - prices[j - 1])
                if spread > current_max_spread:
                    current_max_spread = spread
                current_cum_spread += spread
            prev_tick_sign = current_tick_sign

            if current_tick_sign == 1:
                current_tics_buy += 1
                current_volume_buy += volumes[j]
                current_dollars_buy += prices[j] * volumes[j]
                # Cumulative values
                current_cum_ticks += 1
                current_cum_volumes += volumes[j]
                current_cum_dollars += prices[j] * volumes[j]
            elif current_tick_sign == -1:
                current_tics_sell += 1
                current_volume_sell += volumes[j]
                current_dollars_sell += prices[j] * volumes[j]
                # Cumulative values
                current_cum_ticks -= 1
                current_cum_volumes -= volumes[j]
                current_cum_dollars -= prices[j] * volumes[j]
            else:
                continue

            # Update the cumulative min and max values
            cum_ticks_max[i] = max(cum_ticks_max[i], current_cum_ticks)
            cum_ticks_min[i] = min(cum_ticks_min[i], current_cum_ticks)
            cum_volumes_max[i] = max(cum_volumes_max[i], current_cum_volumes)
            cum_volumes_min[i] = min(cum_volumes_min[i], current_cum_volumes)
            cum_dollars_max[i] = max(cum_dollars_max[i], current_cum_dollars)
            cum_dollars_min[i] = min(cum_dollars_min[i], current_cum_dollars)

        ticks_buy[i] = current_tics_buy
        ticks_sell[i] = current_tics_sell
        volume_buy[i] = current_volume_buy
        volume_sell[i] = current_volume_sell
        dollars_buy[i] = current_dollars_buy
        dollars_sell[i] = current_dollars_sell
        max_spread[i] = current_max_spread
        mean_spread[i] = current_cum_spread / (current_tics_buy + current_tics_sell)

    return (
        ticks_buy, ticks_sell,
        volume_buy, volume_sell,
        dollars_buy, dollars_sell,
        mean_spread, max_spread,
        cum_ticks_min, cum_ticks_max,
        cum_volumes_min, cum_volumes_max,
        cum_dollars_min, cum_dollars_max
    )


@njit(nogil=True, parallel=True)
def comp_bar_trade_size_features(
        amounts: NDArray[np.float64],
        theta: NDArray[np.float64],
        bar_close_indices: NDArray[np.int64],
        theta_mult: float
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
    """
    Compute the size distribution features for each bar, including the mean, 95 percentile, pct_block relative to thehta and size_gini.
    Are there large trade block prints in the bar?

    :param amounts: Array of trade amounts (raw trade sizes).
    :param theta: The typical trade size (e.g., 30 day rolling median trade size).
    :param bar_close_indices: Indices marking the end of each bar.
    :param theta_mult: Multiplier for theta to define the block size threshold. (eg. 5 times the median trade size)
    :returns: A tuple containing:

        - mean_size_rel: Mean trade size relative to theta per bar: log1p(mean_size / theta)
        - size_95_rel: 95th percentile of trade sizes per bar relative to theta: log1p(size_95 / theta)
        - pct_block: Percentage of trades that are larger than theta per bar: SUM( size_i [ size_i>theta ] / volume )
        - size_gini: Gini coefficient of trade sizes per bar.
    """
    if len(theta) != len(bar_close_indices) - 1:
        raise ValueError("Theta should match the the number of bars (len(bar_close_indices) - 1).")

    n_bars = len(bar_close_indices) - 1
    mean_size_rel = np.full(n_bars, np.nan, dtype=np.float32)
    size_95_rel = np.full(n_bars, np.nan, dtype=np.float32)
    pct_block = np.full(n_bars, np.nan, dtype=np.float32)
    size_gini = np.full(n_bars, np.nan, dtype=np.float32)


    for i in prange(n_bars):
        start = bar_close_indices[i] + 1  # Start from the next trade (start=previous bar close)
        end = bar_close_indices[i + 1]

        # Empty bar guard
        if start > end: continue

        if theta[i] == 0.0: continue
        thr = theta[i] * theta_mult  # Block size threshold

        amounts_bar = amounts[start:end + 1]  # End inclusive
        mean_size_rel[i] = np.log1p(np.mean(amounts_bar) / thr)
        size_95_rel[i] = np.log1p(np.percentile(amounts_bar, 95) / thr)

        total_volume = amounts_bar.sum()
        if total_volume == 0:
            continue

        # Calculate pct_block: Percentage of trades larger than block size threshold
        block_volume = 0.0
        for amount in amounts_bar:
            if amount > thr:
                block_volume += amount
        pct_block[i] = block_volume / total_volume

        # Calculate Gini coefficient for trade sizes
        if amounts_bar.size == 1:
            size_gini[i] = 0.0
        else:
            size_gini[i] = 1.0 - np.sum((amounts_bar / total_volume) ** 2)

    return mean_size_rel, size_95_rel, pct_block, size_gini


@njit(nogil=True, parallel=False)  # Currently not parallelizable due to dynamic list usage
def comp_bar_footprints(
    prices: NDArray[np.float64],
    amounts: NDArray[np.float64],
    bar_close_indices: NDArray[np.int64],
    trade_sides: NDArray[np.int8],
    price_tick_size: float,
    bar_lows: NDArray[np.float64],
    bar_highs: NDArray[np.float64],
    imbalance_factor: float
) -> tuple[
    NumbaList[NDArray[np.int32]],
    NumbaList[NDArray[np.float32]], NumbaList[NDArray[np.float32]],
    NumbaList[NDArray[np.int32]], NumbaList[NDArray[np.int32]],
    NumbaList[NDArray[np.bool_]], NumbaList[NDArray[np.bool_]],
    NDArray[np.uint16], NDArray[np.uint16], NDArray[np.int32],
    NDArray[np.int16], NDArray[np.float64], NDArray[np.float64]
]:
    """
    Compute the footprint features for each bar, including buy/sell volumes and imbalances per price level.
    The price levels are calculated in (integer) price tick units to eliminate floating point errors.

    :param prices: Trade prices.
    :param amounts: Trade amounts.
    :param bar_close_indices: Indices marking the end of each bar.
    :param trade_sides: The side information of the market order (1 for market buy, -1 for market sell).
    :param price_tick_size: Tick size used for price level quantization.
    :param bar_lows: Lowest price per bar.
    :param bar_highs: Highest price per bar.
    :param imbalance_factor: Multiplier threshold for detecting imbalance.
    :returns: Tuple containing:

        - price_levels: List of price level arrays per bar.
        - buy_volumes: List of buy volumes per price level.
        - sell_volumes: List of sell volumes per price level.
        - buy_ticks: List of buy ticks per price level.
        - sell_ticks: List of sell ticks per price level.
        - buy_imbalances: List of boolean arrays indicating buy imbalances.
        - sell_imbalances: List of boolean arrays indicating sell imbalances.
        - buy_imbalances_sum: Total number of buy imbalances per bar.
        - sell_imbalances_sum: Total number of sell imbalances per bar.
        - cot_price_levels: Price level with highest total volume per bar.
        - imb_max_run_signed: Longest signed imbalance run for each bar.
        - vp_skew: Volume profile skew for each bar (positive = buy pressure above VWAP).
        - vp_gini: Volume profile Gini coefficient for each bar.
    """
    # TODO: [IDEA] New data structure; Preallocate Flat Arrays and Indices -> This enables parallelization
    n_bars = len(bar_close_indices) - 1

    # Define dynamic lists
    price_levels = NumbaList()
    buy_volumes = NumbaList()
    sell_volumes = NumbaList()
    buy_ticks = NumbaList()
    sell_ticks = NumbaList()

    buy_imbalances = NumbaList()
    sell_imbalances = NumbaList()

    # Initialize cumulative imbalances
    buy_imbalances_sum = np.zeros(n_bars, dtype=np.uint16)
    sell_imbalances_sum = np.zeros(n_bars, dtype=np.uint16)
    cot_price_levels = np.zeros(n_bars, dtype=np.int32)
    imb_max_run_signed_arr = np.zeros(n_bars, dtype=np.int16)
    vp_skew_arr = np.zeros(n_bars, dtype=np.float64)
    vp_gini_arr = np.zeros(n_bars, dtype=np.float64)


    for i in range(n_bars):
        start = bar_close_indices[i] + 1  # Start from the next trade (start=previous bar close)
        end = bar_close_indices[i + 1]

        # Examine current bar price levels
        low = int(round(bar_lows[i] / price_tick_size))
        high = int(round(bar_highs[i] / price_tick_size))
        n_levels = high - low + 1

        # Initialize price levels and volumes
        price_levels_i = np.arange(low, high + 1, dtype=np.int32)
        buy_volumes_i = np.zeros(n_levels, dtype=np.float32)
        sell_volumes_i = np.zeros(n_levels, dtype=np.float32)
        buy_ticks_i = np.zeros(n_levels, dtype=np.int32)
        sell_ticks_i = np.zeros(n_levels, dtype=np.int32)

        # Start aggregating the footprint data (start exclusive, end inclusive)
        for j in range(start, end + 1):
            price = float(prices[j])
            tick_direction = trade_sides[j]
            price = int(round(price / price_tick_size))
            amount = amounts[j]

            # Update the price levels
            # price_level_idx = np.searchsorted(price_levels_i, price)
            price_level_idx = price - low

            # Cumulate the volumes and ticks
            if 0 <= price_level_idx < n_levels:
                if tick_direction == 1:
                    buy_volumes_i[price_level_idx] += amount
                    buy_ticks_i[price_level_idx] += 1
                elif tick_direction == -1:
                    sell_volumes_i[price_level_idx] += amount
                    sell_ticks_i[price_level_idx] += 1
            else:
                raise ValueError("Something went wrong! Invalid price level index!")

        # Append bar's footprint data to the dynamic lists
        price_levels.append(price_levels_i)
        buy_volumes.append(buy_volumes_i)
        sell_volumes.append(sell_volumes_i)
        buy_ticks.append(buy_ticks_i)
        sell_ticks.append(sell_ticks_i)

        # Calculate the footprint features:
        # buy imbalances, sell imbalances, imb_max_run_signed, COT price level, vp_skew, vp_gini
        (buy_imbalances_i, sell_imbalances_i, imb_max_run_signed,
         cot_price_level, vp_skew, vp_gini) = (
            comp_footprint_features(price_levels_i, buy_volumes_i, sell_volumes_i, imbalance_factor
        ))
        buy_imbalances.append(buy_imbalances_i)
        sell_imbalances.append(sell_imbalances_i)

        # Update cumulative imbalances, COT price level, and other metrics
        buy_imbalances_sum[i] = np.sum(buy_imbalances_i, dtype=np.uint16)
        sell_imbalances_sum[i] = np.sum(sell_imbalances_i, dtype=np.uint16)
        cot_price_levels[i] = cot_price_level
        imb_max_run_signed_arr[i] = imb_max_run_signed
        vp_skew_arr[i] = vp_skew
        vp_gini_arr[i] = vp_gini

    return (
        price_levels,
        buy_volumes, sell_volumes,
        buy_ticks, sell_ticks,
        buy_imbalances, sell_imbalances,
        buy_imbalances_sum, sell_imbalances_sum, cot_price_levels,
        imb_max_run_signed_arr, vp_skew_arr, vp_gini_arr
    )


@njit(nogil=True)
def comp_footprint_features(price_levels, buy_volumes, sell_volumes, imbalance_multiplier):
    """
    Calculate footprint statistics such as buy/sell imbalances and Commitment of Traders (COT) level.

    :param price_levels: Array of int64 tick unit price levels in ascending order.
    :param buy_volumes: Array of buy volumes at each price level.
    :param sell_volumes: Array of sell volumes at each price level.
    :param imbalance_multiplier: Threshold multiplier to detect imbalance.
    :returns: Tuple containing:

        - buy_imbalances: Boolean array where True indicates buy imbalance at the level.
        - sell_imbalances: Boolean array where True indicates sell imbalance at the level.
        - imbalance_max_run_signed: Longest  signed imbalance run (number of consecutive imbalanced level)
        - cot_price_level: Price level with the highest total volume.
        - vp_skew: Volume profile skew relative to vwap (positive = buy pressure above VWAP).
        - vp_gini: Volume profile Gini coefficient (0 = concentrated, →1 = even distribution).
    """
    n_levels = len(price_levels)
    buy_imbalances = np.zeros(n_levels, dtype=np.bool_)
    sell_imbalances = np.zeros(n_levels, dtype=np.bool_)

    # Ascending Bid-Ask Table         Descending Bid-Ask Table (Conventional)
    # -----------------                 -----------------
    # | L | Sell| Buy |                 | L | Sell| Buy |
    # |---|-----|-----|                 |---|-----|-----|
    # | 0 |   0 |   1 |                 | 3 |  10 |   0 |
    # |---|-----|-----|                 |---|-----|-----|
    # | 1 |   2 |  56 |                 | 2 | 181 |  15 |
    # |---|-----|-----|                 |---|-----|-----|
    # | 2 | 181 |  15 |                 | 1 |   2 |  56 |
    # |---|-----|-----|                 |---|-----|-----|
    # | 3 |  10 |   0 |                 | 0 |   0 |   1 |
    # -----------------                 -----------------
    # In the conventional table the l-th sell (bid) level corresponds to the (l-1)-th buy (ask) level;
    # the l-th buy (ask) level corresponds to the (l+1)-th sell (bid) level
    #
    # In the ascending table it is reversed and the l-th sell (bid) level corresponds to the (l+1)-th buy (ask) level;
    # the l-th buy (ask) level corresponds to the (l-1)-th sell (bid) level

    if n_levels > 1:
        # Edge levels cannot be imbalanced by definition (we cannot compare it to the next diagonal level)
        sell_imbalances[:-1] = sell_volumes[:-1] > (buy_volumes[1:] * imbalance_multiplier)
        buy_imbalances[1:] = buy_volumes[1:] > (sell_volumes[:-1] * imbalance_multiplier)

    # ---------- longest signed run ----------
    max_run = 0
    max_sign = 0  # +1 buy, -1 sell
    run = 0
    run_sign = 0

    for i in range(n_levels):
        sign = 1 if buy_imbalances[i] else (-1 if sell_imbalances[i] else 0)

        if sign != 0 and sign == run_sign:
            run += 1
        elif sign != 0:  # start new run
            run = 1
            run_sign = sign
        else:  # level with no imbalance
            run = 0
            run_sign = 0

        if run > max_run:
            max_run = run
            max_sign = run_sign

    imb_max_run_signed = max_run * max_sign  # int16

    # ---------- COT & VP stats ----------
    total_volumes = buy_volumes + sell_volumes
    sum_total_volume = total_volumes.sum()

    # Calculate Center of Trades (COT)
    highest_cot_idx = np.argmax(total_volumes)
    cot_price_level = price_levels[highest_cot_idx]

    # Calculate VWAP for skew computation
    vp_skew = 0.0
    vp_gini = 0.0

    if sum_total_volume > 0 and n_levels > 0:
        # Calculate VWAP
        vwap = np.sum(price_levels * total_volumes) / sum_total_volume

        # 1. Calculate volume profile skew
        # Using the formula: skew = Σ(p-vwap) * vol) / tot_vol
        price_deviation = price_levels - vwap
        vp_skew = np.dot(price_deviation, total_volumes) / sum_total_volume

        # 2. Calculate volume profile Gini coefficient
        # Using the formula: gini = 1 - Σ((vol_i / tot_vol) ** 2)
        volume_proportions = total_volumes / sum_total_volume
        vp_gini = 1.0 - np.sum(volume_proportions ** 2)

    return buy_imbalances, sell_imbalances, imb_max_run_signed, cot_price_level, vp_skew, vp_gini
