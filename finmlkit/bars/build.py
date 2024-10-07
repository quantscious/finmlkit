"""
This module contains the functions to build candlestick bars and other intra-bar features
(i.e. directional features, footprints) from raw trades data using the indexer functions outputs
defined in the logic module.
"""
import numpy as np
from numba import njit
from numpy.typing import NDArray
from numba import prange
from .utils import comp_trade_side_vector, footprint_to_dataframe
from dataclasses import dataclass
from typing import Union, Optional, List, Dict, Tuple
import pandas as pd
import datetime as dt
import numba as nb
from abc import ABC, abstractmethod


@dataclass
class FootprintData:
    """
    Footprint data container for the dynamic memory footprint calculations.
    All list attributes are initialized as numpy arrays with dtype=object to ensure they are serializable.
    """
    # Data attributes
    bar_timestamps: np.ndarray      # 1D int64 array
    price_levels: Union[np.ndarray, list[np.array]]        # Array of 1D int32 arrays (price levels in price tick units)
    price_tick:   float               # Price tick size (float)
    buy_volumes:  Union[np.ndarray, list[np.array]]         # Array of 1D float32 arrays
    sell_volumes: Union[np.ndarray, list[np.array]]         # Array of 1D float32 arrays
    buy_ticks:    Union[np.ndarray, list[np.array]]         # Array of 1D int32 arrays
    sell_ticks:   Union[np.ndarray, list[np.array]]         # Array of 1D int32 arrays
    buy_imbalances:  Union[np.ndarray, list[np.array]]      # Array of 1D bool arrays
    sell_imbalances: Union[np.ndarray, list[np.array]]      # Array of 1D bool arrays

    # Additional attributes
    cot_price_levels: Optional[np.ndarray] = None  # 1D int32 array
    sell_imbalances_sum: Optional[np.ndarray] = None  # 1D uint16 array
    buy_imbalances_sum: Optional[np.ndarray] = None  # 1D uint16 array

    # Private attributes
    _datetime_index: pd.Series = None  # DatetimeIndex for date time slicing (will be set in __post_init__)

    def __post_init__(self):
        # Convert bar_timestamps to pandas DatetimeIndex for easier slicing
        # timestamps are in nanoseconds
        self._datetime_index = pd.to_datetime(self.bar_timestamps, unit='ns')


    def __len__(self) -> int:
        """
        Returns the number of data points in the FootprintData object.
        """
        return len(self.bar_timestamps)

    def __repr__(self) -> str:
        """
        String representation for debugging and logging.
        """
        additional_info = {
            'cot_price_levels': 'present' if self.cot_price_levels is not None else 'missing',
            'sell_imbalances_sum': 'present' if self.sell_imbalances_sum is not None else 'missing',
            'buy_imbalances_sum': 'present' if self.buy_imbalances_sum is not None else 'missing'
        }

        # Check if all arrays are of the same type
        array_types = {type(getattr(self, attr)).__name__ for attr in [
            'price_levels', 'buy_volumes', 'sell_volumes',
            'buy_ticks', 'sell_ticks', 'buy_imbalances', 'sell_imbalances'
        ]}

        if len(array_types) == 1:
            array_type = array_types.pop()
            type_message = array_type
        else:
            type_message = f"Warning: Arrays have mixed types: {', '.join(array_types)}"

        # Calculate total memory usage
        total_memory_usage = self.memory_usage()

        return (
            f"FootprintData:\n"
            f"  Number of Bars: {len(self)}\n"
            f"  Price Tick: {self.price_tick}\n"
            f"  Date Range: {self._datetime_index[0]} to {self._datetime_index[-1]}\n"
            f"  Array Types: {type_message}\n"
            f"  Optional Attributes:\n"
            f"    COT Price Levels: {additional_info['cot_price_levels']}\n"
            f"    Sell Imbalances Sum: {additional_info['sell_imbalances_sum']}\n"
            f"    Buy Imbalances Sum: {additional_info['buy_imbalances_sum']}\n"
            f"  Total Memory Usage: {total_memory_usage:.3f} MB\n"
        )

    def __getitem__(self, key) -> 'FootprintData':
        """
        Enable slicing of the FootprintData object.
        Args:
            key: The slice or integer index to access specific data.
        Returns:
            A new FootprintData object with the sliced data.
        """
        if isinstance(key, slice):
            if isinstance(key.start, (str, dt.datetime)) and isinstance(key.stop, (str, dt.datetime)):
                # Use pandas indexing with datetime slice
                start_idx, end_idx = self._datetime_index.slice_locs(start=key.start, end=key.stop)
                return self[start_idx:end_idx]

            # Regular numeric slicing
            return FootprintData(
                bar_timestamps=self.bar_timestamps[key],
                price_levels=self.price_levels[key],
                price_tick=self.price_tick,
                buy_volumes=self.buy_volumes[key],
                sell_volumes=self.sell_volumes[key],
                buy_ticks=self.buy_ticks[key],
                sell_ticks=self.sell_ticks[key],
                buy_imbalances=self.buy_imbalances[key],
                sell_imbalances=self.sell_imbalances[key],
                cot_price_levels=self.cot_price_levels[key] if self.cot_price_levels is not None else None,
                sell_imbalances_sum=self.sell_imbalances_sum[key] if self.sell_imbalances_sum is not None else None,
                buy_imbalances_sum=self.buy_imbalances_sum[key] if self.buy_imbalances_sum is not None else None
            )
        else:
            raise TypeError("Invalid argument type. Expected a slice.")

    @classmethod
    def from_numba(cls, data: Tuple, price_tick: float) -> 'FootprintData':
        """
        Initialize the FootprintData data container from the output of the calculate_footprint_dynamic function.
        (The output is a numba list which we cast to numpy arrays for serialization and saving purposes.)
        Args:
            data: Output of the calculate_footprint_dynamic function, containing array of numpy arrays.
            price_tick: The price tick size.
        """
        instance = cls(
            bar_timestamps=np.array(data[0], dtype=np.int64),
            price_levels=np.array(data[1], dtype=object),
            price_tick=price_tick,
            buy_volumes=np.array(data[2], dtype=object),
            sell_volumes=np.array(data[3], dtype=object),
            buy_ticks=np.array(data[4], dtype=object),
            sell_ticks=np.array(data[5], dtype=object),
            buy_imbalances=np.array(data[6], dtype=object),
            sell_imbalances=np.array(data[7], dtype=object),
            buy_imbalances_sum=np.array(data[8], dtype=np.int16),
            sell_imbalances_sum=np.array(data[9], dtype=np.int16),
            cot_price_levels=np.array(data[10], dtype=np.int32)
        )

        # Validate the data
        if not instance.is_valid():
            raise ValueError("Inconsistent data length in the FootprintData container!")

        return instance

    @classmethod
    def from_dict(cls, data: Dict) -> 'FootprintData':
        """
        Initialize the FootprintData data container from a dictionary.
        Args:
            data: Dictionary containing the footprint data;
                    keys: bar_timestamps, price_levels, buy_volumes, sell_volumes,
                          buy_ticks, sell_ticks, buy_imbalances, sell_imbalances, price_tick
        """
        instance = cls(
            bar_timestamps=data['bar_timestamps'],
            price_levels=data['price_levels'],
            price_tick=data['price_tick'],
            buy_volumes=data['buy_volumes'],
            sell_volumes=data['sell_volumes'],
            buy_ticks=data['buy_ticks'],
            sell_ticks=data['sell_ticks'],
            buy_imbalances=data['buy_imbalances'],
            sell_imbalances=data['sell_imbalances']
        )

        # Validate the data
        if not instance.is_valid():
            raise ValueError("Inconsistent data length in the FootprintData container!")

        return instance

    def get_df(self):
        """
        Convert the footprint data to a pandas DataFrame format.
        Returns: footprint dataframe
        """
        df = footprint_to_dataframe(self.bar_timestamps, self.price_levels, self.buy_volumes,
                                    self.sell_volumes, self.buy_ticks, self.sell_ticks,
                                    self.buy_imbalances, self.sell_imbalances, self.price_tick)
        return df

    def cast_to_numba_list(self):
        """
        Cast the footprint data in-place to a numba list for numba calculations.
        (Numba does not support numpy arrays with dtype=object, we have to cast them to numba lists.)
        """
        self.price_levels = nb.typed.List(self.price_levels)
        self.buy_volumes = nb.typed.List(self.buy_volumes)
        self.sell_volumes = nb.typed.List(self.sell_volumes)
        self.buy_ticks = nb.typed.List(self.buy_ticks)
        self.sell_ticks = nb.typed.List(self.sell_ticks)
        self.buy_imbalances = nb.typed.List(self.buy_imbalances)
        self.sell_imbalances = nb.typed.List(self.sell_imbalances)

    def cast_to_numpy(self):
        """
        Cast the footprint data in-place to numpy arrays object for general processing and serialization.
        """
        self.price_levels = np.array(self.price_levels, dtype=object)
        self.buy_volumes = np.array(self.buy_volumes, dtype=object)
        self.sell_volumes = np.array(self.sell_volumes, dtype=object)
        self.buy_ticks = np.array(self.buy_ticks, dtype=object)
        self.sell_ticks = np.array(self.sell_ticks, dtype=object)
        self.buy_imbalances = np.array(self.buy_imbalances, dtype=object)
        self.sell_imbalances = np.array(self.sell_imbalances, dtype=object)

    def memory_usage(self) -> float:
        """
        Calculate total memory usage of the FootprintData object by iterating through its elements.
        """
        from pympler import asizeof  # memory profiler

        total_memory = 0
        attributes = [
            'bar_timestamps', 'price_levels', 'buy_volumes',
            'sell_volumes', 'buy_ticks', 'sell_ticks',
            'buy_imbalances', 'sell_imbalances',
            'cot_price_levels', 'sell_imbalances_sum', 'buy_imbalances_sum'
        ]

        for attr in attributes:
            array = getattr(self, attr)
            if isinstance(array, np.ndarray) or isinstance(array, list) or isinstance(array, nb.typed.List):
                # Use pympler to measure the size of each element
                total_memory += sum(asizeof.asizeof(item) for item in array if item is not None)

        return total_memory / (1024 ** 2)  # Convert to MB

    def is_valid(self) -> bool:
        # Check for consistent lengths and types of all attributes
        expected_length = len(self.bar_timestamps)
        attributes = [
            self.price_levels, self.buy_volumes, self.sell_volumes,
            self.buy_ticks, self.sell_ticks, self.buy_imbalances, self.sell_imbalances
        ]
        for attr in attributes:
            if len(attr) != expected_length:
                return False
        return True


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

    def build_ohlcv(self) -> pd.DataFrame:
        """
        Build the bar features using the generated indices and raw trades data.
        Returns
        -------
        pd.DataFrame
            A dataframe containing the OHLCV + VWAP features.
        """
        pass

    def build_directional_features(self) -> pd.DataFrame:
        """
        Build the directional features using the generated indices and raw trades data.
        Returns
        -------
        pd.DataFrame
            A dataframe containing the directional features:
            ticks_buy, ticks_sell, volume_buy, volume_sell, dollars_buy, dollars_sell, max_spread,
            cum_volumes_min, cum_volumes_max, cum_dollars_min, cum_dollars_max.
        """
        pass

    def build_footprints(self) -> FootprintData:
        """
        Build the footprint data using the generated indices and raw trades data.
        Returns
        -------
        FootprintData
            A FootprintData object containing the footprint data.
        """
        pass


@njit(nopython=True, nogil=True, parallel=True)
def comp_bar_ohlcv(
    prices: NDArray[np.float64],
    volumes: NDArray[np.float64],
    bar_open_indices: NDArray[np.int64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float32], NDArray[np.float64]]:
    """
    Build the candlestick bars from raw trades data based in bar open indices.

    Parameters
    ----------
    prices : np.array(np.float64)
        Raw sorted **trades data** prices.
    volumes : np.array(np.float64)
        Raw sorted **trades data** volumes.
    bar_open_indices : np.array(np.int64)
        Bar open indices calculated with a bar indexer function.

    Returns
    -------
    tuple (np.array(np.float64), np.array(np.float64), np.array(np.float64), np.array(np.float64), np.array(np.float32), np.array(np.float64))
        A tuple containing the bars' open, high, low, close prices, volumes and vwap. OHLCV + VWAP.
    """
    n_bars = len(bar_open_indices) - 1  # The last open index determines the last bars' close
    bar_high = np.zeros(n_bars, dtype=np.float64)
    bar_low = np.zeros(n_bars, dtype=np.float64)
    bar_open = np.zeros(n_bars, dtype=np.float64)
    bar_close = np.zeros(n_bars, dtype=np.float64)
    bar_volume = np.zeros(n_bars, dtype=np.float64)
    bar_vwap = np.zeros(n_bars, dtype=np.float32)
    bar_dollar = np.zeros(n_bars, dtype=np.float64)

    for i in prange(n_bars):
        start = bar_open_indices[i]
        end = bar_open_indices[i + 1]

        # Handle empty bars
        if start == end:
            last_price = prices[start - 1]
            bar_open[i] = last_price
            bar_close[i] = last_price
            bar_high[i] = last_price
            bar_low[i] = last_price
            bar_volume[i] = 0.0
            bar_dollar[i] = 0.0
            bar_vwap[i] = 0.0
            continue

        # Initialize variables for this bar
        high_price = prices[start]
        low_price = prices[start]
        total_volume = 0.0
        total_dollar = 0.0

        # Iterate over trades in the current bar
        for j in range(start, end):
            price = prices[j]
            volume = volumes[j]

            if price > high_price:
                high_price = price
            if price < low_price:
                low_price = price

            total_volume += volume
            total_dollar += price * volume

        bar_open[i] = prices[start]
        bar_close[i] = prices[end - 1]
        bar_high[i] = high_price
        bar_low[i] = low_price
        bar_volume[i] = total_volume
        bar_dollar[i] = total_dollar
        bar_vwap[i] = total_dollar / total_volume if total_volume > 0 else 0.0

    return bar_open, bar_high, bar_low, bar_close, bar_volume, bar_vwap



@njit(nopython=True, nogil=True, parallel=True)
def comp_bar_directional_features(
    prices: NDArray[np.float64],
    volumes: NDArray[np.float64],
    bar_open_indices: NDArray[np.int64]
) -> tuple[
    NDArray[np.int64], NDArray[np.int64],
    NDArray[np.float32], NDArray[np.float32],
    NDArray[np.float32], NDArray[np.float32],
    NDArray[np.float32],
    NDArray[np.float64], NDArray[np.float64],
    NDArray[np.float32], NDArray[np.float32],
    NDArray[np.float32], NDArray[np.float32]
]:
    """
    Compute directional bar features like sell/buy volumes etc... for each bar.

    Parameters
    ----------
    prices : np.array(np.float64)
        Raw trades prices.
    volumes : np.array(np.float64)
        Raw trades volumes.
    bar_open_indices : np.array(np.int64)
        Bar open indices in the raw trades timestamps.

    Returns
    -------
    tuple
        A tuple containing the bars' ticks_buy, ticks_sell, volume_buy, volume_sell, dollars_buy, dollars_sell, max_spread, cum_volumes_min, cum_volumes_max, cum_dollars_min, cum_dollars_max.
    """
    n_bars = len(bar_open_indices) - 1
    ticks_buy = np.zeros(n_bars, dtype=np.int64)
    ticks_sell = np.zeros(n_bars, dtype=np.int64)
    volume_buy = np.zeros(n_bars, dtype=np.float32)
    volume_sell = np.zeros(n_bars, dtype=np.float32)
    dollars_buy = np.zeros(n_bars, dtype=np.float32)
    dollars_sell = np.zeros(n_bars, dtype=np.float32)
    max_spread = np.zeros(n_bars, dtype=np.float32)

    # Initialize cumulative min and max arrays with appropriate values
    cum_ticks_min = np.full(n_bars, 1e9, dtype=np.int64)            # inf (large value)
    cum_ticks_max = np.full(n_bars, -1e9, dtype=np.int64)                   # -inf (small value)
    cum_volumes_min = np.full(n_bars, 1e9, dtype=np.float32)
    cum_volumes_max = np.full(n_bars, -1e9, dtype=np.float32)
    cum_dollars_min = np.full(n_bars, 1e9, dtype=np.float32)
    cum_dollars_max = np.full(n_bars, -1e9, dtype=np.float32)

    # 1.) Obtaining direction information
    # Calculate trade side for each trade
    trade_sides = comp_trade_side_vector(prices)

    # 2.) Compute the bar directional features
    for i in prange(n_bars):
        start = bar_open_indices[i]
        end = bar_open_indices[i + 1]

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

        # Initialize previous tick sign for spread calculation
        if end > start:
            prev_tick_sign = trade_sides[start]
        else:
            prev_tick_sign = 0  # Default value if no trades in bar

        for j in range(start, end):
            current_tick_sign = trade_sides[j]

            # Calculate the spread between buy and sell prices
            if current_tick_sign != prev_tick_sign:
                spread = abs(prices[j] - prices[j - 1])
                if spread > current_max_spread:
                    current_max_spread = spread
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

    return (
        ticks_buy, ticks_sell,
        volume_buy, volume_sell,
        dollars_buy, dollars_sell,
        max_spread,
        cum_ticks_min, cum_ticks_max,
        cum_volumes_min, cum_volumes_max,
        cum_dollars_min, cum_dollars_max
    )

@njit(nopython=True, nogil=True)
def comp_bar_footprints(
    prices: NDArray[np.float64],
    volumes: NDArray[np.float64],
    bar_open_indices: NDArray[np.int64]
):
    pass
