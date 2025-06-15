"""
This module contains the functions to build candlestick bar and other intra-bar features
(i.e. directional features, footprints) from raw trades data using the indexer functions outputs
defined in the logic module.
"""
import numpy as np
from numba import njit
from numpy.typing import NDArray
from numba import prange
from typing import Tuple, Optional
import pandas as pd
from numba.typed import List as NumbaList
from abc import ABC, abstractmethod
from numpy.typing import NDArray

from .data_model import FootprintData
from .utils import comp_price_tick_size, comp_trade_side
from .data_model import TradesData

from finmlkit.utils.log import get_logger
logger = get_logger(__name__)


class BarBuilderBase(ABC):
    """
    This class provides a template for generating bar from raw trades data.
    """

    def __init__(self,trades: TradesData):
        """
        Initialize the bar builder with raw trades data.

        :param trades: DataFrame containing raw trades data containing 'timestamp'/'time', 'price', and 'amount'/'qty'.
            If 'is_buyer_maker' is present, it indicates the trade side otherwise it is inferred.
        """
        self.trades_df = trades.data

        self._close_ts:      Optional[NDArray[np.int64]] = None
        self._close_indices: Optional[NDArray[np.int64]] = None
        self._highs:        Optional[NDArray[np.float64]] = None
        self._lows:         Optional[NDArray[np.float64]] = None

    def __str__(self):
        return (f"Class: {self.__class__.__name__} with members:\n"
                f"{[f"{key}: {value}\n" for key, value in self.__dict__.items()]} "
                f"\nRaw trades data:\n{self.trades_df.info()}")

    @abstractmethod
    def _comp_bar_close(self) -> Tuple[NDArray[np.int64], NDArray[np.int64]]:
        """
        Abstract method to generate bar open timestamps and indices.
        :returns: Tuple of open timestamps and their corresponding indices.
        """
        pass

    def _calc_bar_open_values(self):
        """
        Calculate and sets the open timestamps and indices if not already calculated.
        """
        if self._close_indices is None:
            logger.info("Calculating bar open tick indices and timestamps...")
            self._close_ts, self._close_indices = self._comp_bar_close()

    @property
    def close_indices(self) -> Optional[NDArray[np.int64]]:
        """
        Return the bar close indices in the raw trades data.
        :return:
        """
        if self._close_indices is None:
            print("Bar open indices are not calculated yet. Call _calc_bar_open_values() first.")
            return None
        return self._close_indices[1:]


    def build_ohlcv(self) -> pd.DataFrame:
        """
        Build the bar features using the generated indices and raw trades data.
        :returns: A dataframe containing the OHLCV + VWAP features with datetime index corresponding to the bar open timestamps.
        """
        self._calc_bar_open_values()

        ohlcv_tuple = comp_bar_ohlcv(
            self.trades_df['price'].values,
            self.trades_df['amount'].values,
            self._close_indices
        )
        self._highs, self._lows = ohlcv_tuple[1], ohlcv_tuple[2]
        logger.info("OHLCV bar calculated successfully.")

        ohlcv_df = pd.DataFrame({
            'timestamp': self._close_ts[1:],   # Close bar timestamps convention!!
            'open': ohlcv_tuple[0],
            'high': ohlcv_tuple[1],
            'low': ohlcv_tuple[2],
            'close': ohlcv_tuple[3],
            'volume': ohlcv_tuple[4],
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
        self._calc_bar_open_values()

        directional_tuple = comp_bar_directional_features(
            self.trades_df['price'].values,
            self.trades_df['amount'].values,
            self._close_indices,
            self.trades_df['side'].values.astype(np.int8),
        )
        logger.info("Directional features calculated successfully.")

        directional_df = pd.DataFrame({
            'timestamp': self._close_ts[1:],     # Close bar timestamps convention!!
            'ticks_buy': directional_tuple[0],
            'ticks_sell': directional_tuple[1],
            'volume_buy': directional_tuple[2],
            'volume_sell': directional_tuple[3],
            'dollars_buy': directional_tuple[4],
            'dollars_sell': directional_tuple[5],
            'mean_spread': directional_tuple[6],
            'max_spread': directional_tuple[7],
            'cum_volumes_min': directional_tuple[8],
            'cum_volumes_max': directional_tuple[9],
            'cum_dollars_min': directional_tuple[10],
            'cum_dollars_max': directional_tuple[11]
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

        self._calc_bar_open_values()
        if self._highs is None or self._lows is None:
            # We need the bar highs and lows for the footprint calculation
            self.build_ohlcv()

        if price_tick_size is None:
            # Anticipate price tick size
            price_tick_size = comp_price_tick_size(self.trades_df['price'].values)
        logger.info(f"Price tick size is set to: {price_tick_size}")

        # Compute the footprint data
        footprint_data = comp_bar_footprints(
            self.trades_df['price'].values,
            self.trades_df['amount'].values,
            self._close_indices,
            self._close_ts,
            price_tick_size,
            self._lows,
            self._highs,
            imbalance_factor
        )
        logger.info("Footprint data calculated successfully.")

        # Create a FootprintData object with all metrics
        footprint = FootprintData(
            bar_timestamps=footprint_data[0],
            price_levels=footprint_data[1],
            price_tick=price_tick_size,
            buy_volumes=footprint_data[2],
            sell_volumes=footprint_data[3],
            buy_ticks=footprint_data[4],
            sell_ticks=footprint_data[5],
            buy_imbalances=footprint_data[6],
            sell_imbalances=footprint_data[7],
            buy_imbalances_sum=footprint_data[8],
            sell_imbalances_sum=footprint_data[9],
            cot_price_levels=footprint_data[10],
            imb_max_run_signed=footprint_data[11],
            vp_skew=footprint_data[12],
            vp_gini=footprint_data[13]
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
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float32], NDArray[np.float64]]:
    """
    Build the candlestick bar from raw trades data based in bar open indices.
    :param prices: Trade prices.
    :param volumes: Trade volumes.
    :param bar_close_indices: Indices marking the start of each bar.
    :returns: Tuple containing:
        - open: Opening price of each bar.
        - high: Highest price of each bar.
        - low: Lowest price of each bar.
        - close: Closing price of each bar.
        - volume: Total traded volume in each bar.
        - vwap: Volume-weighted average price of each bar.
    """
    # Check the input arrays match in length
    if len(prices) != len(volumes):
        raise ValueError("Prices and volumes arrays must have the same length.")
    if len(bar_close_indices) < 2:
        raise ValueError("Bar open indices must contain at least two elements.")

    n_bars = len(bar_close_indices) - 1  # The last open index determines the last bar's close
    bar_high = np.zeros(n_bars, dtype=np.float64)
    bar_low = np.zeros(n_bars, dtype=np.float64)
    bar_open = np.zeros(n_bars, dtype=np.float64)
    bar_close = np.zeros(n_bars, dtype=np.float64)
    bar_volume = np.zeros(n_bars, dtype=np.float32)
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
            continue

        # Start from the next trade (start=previous bar close)
        start += 1
        # Initialize variables for this bar
        high_price = prices[start]
        low_price = prices[start]
        total_volume = 0.0
        total_dollar = 0.0

        # Iterate over trades in the current bar, inclusive for the last trade (bar close)
        for j in range(start, end + 1):
            price = prices[j]
            volume = volumes[j]

            if price > high_price:
                high_price = price
            if price < low_price:
                low_price = price

            total_volume += volume
            total_dollar += price * volume

        bar_open[i] = prices[start]  # First trade price in the bar exclusive
        bar_close[i] = prices[end]   # Last trade price in the bar inclusive
        bar_high[i] = high_price
        bar_low[i] = low_price
        bar_volume[i] = total_volume
        bar_vwap[i] = total_dollar / total_volume if total_volume > 0 else 0.0

    return bar_open, bar_high, bar_low, bar_close, bar_volume, bar_vwap


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
    :param bar_close_indices: Indices marking the start of each bar.
    :param trade_sides: Trade direction (1 for market buy, -1 for market sell).
    :returns: Tuple containing:
        - **ticks_buy**: Number of buy trades per bar.
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
            prev_tick_sign = trade_sides[start]
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


@njit(nogil=True, parallel=False)
def comp_bar_footprints(
    prices: NDArray[np.float64],
    amounts: NDArray[np.float64],
    bar_close_indices: NDArray[np.int64],
    bar_open_timestamps: NDArray[np.int64],
    price_tick_size: float,
    bar_lows: NDArray[np.float64],
    bar_highs: NDArray[np.float64],
    imbalance_factor: float
) -> tuple[
    NDArray[np.int64],
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
    :param bar_close_indices: Indices marking the start of each bar.
    :param bar_open_timestamps: Nanosecond timestamps marking bar openings.
    :param price_tick_size: Tick size used for price level quantization.
    :param bar_lows: Lowest price per bar.
    :param bar_highs: Highest price per bar.
    :param imbalance_factor: Multiplier threshold for detecting imbalance.
    :returns: Tuple containing:
        - bar_open_timestamps: Timestamps for each bar.
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

    tick_direction = 0
    for i in prange(n_bars):
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
            previous_price = prices[j - 1] if j > 0 else price
            tick_direction = comp_trade_side(price, previous_price, tick_direction)
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
         cot_price_level, vp_skew, vp_gini) = comp_footprint_features(
            price_levels_i, buy_volumes_i, sell_volumes_i, imbalance_factor
        )
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
        bar_open_timestamps[1:],  # Close bar timestamps convention!!
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
        - vp_skew: Volume profile skew (positive = buy pressure above VWAP).
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
    max_sign = 0       # +1 buy, -1 sell
    run = 0
    run_sign = 0

    for i in range(n_levels):
        sign = 1 if buy_imbalances[i] else (-1 if sell_imbalances[i] else 0)

        if sign != 0 and sign == run_sign:
            run += 1
        elif sign != 0:          # start new run
            run = 1
            run_sign = sign
        else:                    # level with no imbalance
            run = 0
            run_sign = 0

        if run > max_run:
            max_run  = run
            max_sign = run_sign

    imb_max_run_signed = max_run * max_sign      # int16

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
        vp_skew = np.sum(price_deviation * total_volumes) / sum_total_volume

        # 2. Calculate volume profile Gini coefficient
        # Using the formula: gini = 1 - Σ((vol_i / tot_vol) ** 2)
        volume_proportions = total_volumes / sum_total_volume
        vp_gini = 1.0 - np.sum(volume_proportions ** 2)

    return buy_imbalances, sell_imbalances, imb_max_run_signed, cot_price_level, vp_skew, vp_gini
