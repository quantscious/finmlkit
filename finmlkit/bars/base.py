"""
This module contains the functions to build candlestick bars and other intra-bar features
(i.e. directional features, footprints) from raw trades data using the indexer functions outputs
defined in the logic module.
"""
import numpy as np
from numba import njit
from numpy.typing import NDArray
from numba import prange
from .utils import comp_trade_side_vector, footprint_to_dataframe, comp_price_tick_size, comp_trade_side
from dataclasses import dataclass
from typing import Union, Optional, List, Dict, Tuple
import pandas as pd
import datetime as dt
from numba.typed import List as NumbaList
from abc import ABC, abstractmethod
from finmlkit.utils.log import logger


@dataclass
class FootprintData:
    """
    FootprintData is a container for dynamic memory footprint calculations.

    Parameters
    ----------
    bar_timestamps : NDArray[np.int64]
        1D array of bar timestamps in nanoseconds.
    price_levels : Union[NDArray[NDArray[np.int32]], NumbaList[NDArray[np.int32]]]
        Array of 1D int32 arrays representing price levels in price tick units.
    price_tick : float
        Price tick size.
    buy_volumes : Union[NDArray[NDArray[np.float32]], NumbaList[NDArray[np.float32]]]
        Array of 1D float32 arrays representing buy volumes.
    sell_volumes : Union[NDArray[NDArray[np.float32]], NumbaList[NDArray[np.float32]]]
        Array of 1D float32 arrays representing sell volumes.
    buy_ticks : Union[NDArray[NDArray[np.int32]], NumbaList[NDArray[np.int32]]]
        Array of 1D int32 arrays representing buy ticks.
    sell_ticks : Union[NDArray[NDArray[np.int32]], NumbaList[NDArray[np.int32]]]
        Array of 1D int32 arrays representing sell ticks.
    buy_imbalances : Union[NDArray[NDArray[np.bool_]], NumbaList[NDArray[np.bool_]]]
        Array of 1D bool arrays representing buy imbalances.
    sell_imbalances : Union[NDArray[NDArray[np.bool_]], NumbaList[NDArray[np.bool_]]]
        Array of 1D bool arrays representing sell imbalances.
    cot_price_levels : Optional[NDArray[np.int32]], optional
        1D int32 array of COT price levels, by default None.
    sell_imbalances_sum : Optional[NDArray[np.uint16]], optional
        1D uint16 array of summed sell imbalances, by default None.
    buy_imbalances_sum : Optional[NDArray[np.uint16]], optional
        1D uint16 array of summed buy imbalances, by default None.

    Attributes
    ----------
    bar_timestamps : NDArray[np.int64]
        Timestamps of the bars.
    price_levels : Union[NDArray[NDArray[np.int32]], NumbaList[NDArray[np.int32]]]
        Price levels in price tick units.
    price_tick : float
        Price tick size.
    buy_volumes : Union[NDArray[NDArray[np.float32]], NumbaList[NDArray[np.float32]]]
        Buy volumes.
    sell_volumes : Union[NDArray[NDArray[np.float32]], NumbaList[NDArray[np.float32]]]
        Sell volumes.
    buy_ticks : Union[NDArray[NDArray[np.int32]], NumbaList[NDArray[np.int32]]]
        Buy ticks.
    sell_ticks : Union[NDArray[NDArray[np.int32]], NumbaList[NDArray[np.int32]]]
        Sell ticks.
    buy_imbalances : Union[NDArray[NDArray[np.bool_]], NumbaList[NDArray[np.bool_]]]
        Buy imbalances.
    sell_imbalances : Union[NDArray[NDArray[np.bool_]], NumbaList[NDArray[np.bool_]]]
        Sell imbalances.
    cot_price_levels : Optional[NDArray[np.int32]]
        COT price levels.
    sell_imbalances_sum : Optional[NDArray[np.uint16]]
        Summed sell imbalances.
    buy_imbalances_sum : Optional[NDArray[np.uint16]]
        Summed buy imbalances.
    _datetime_index : pd.DatetimeIndex
        Datetime index for date time slicing.
    """
    # Data attributes
    bar_timestamps: NDArray[np.int64]  # 1D int64 array
    price_tick: float  # Price tick size (float)
    price_levels: Union[NDArray[NDArray[np.int32]], NumbaList[NDArray[np.int32]]]     # Array of 1D int32 arrays (price levels in price tick units)
    buy_volumes: Union[NDArray[NDArray[np.float32]], NumbaList[NDArray[np.float32]]]  # Array of 1D float32 arrays
    sell_volumes: Union[NDArray[NDArray[np.float32]], NumbaList[NDArray[np.float32]]] # Array of 1D float32 arrays
    buy_ticks: Union[NDArray[NDArray[np.int32]], NumbaList[NDArray[np.int32]]]        # Array of 1D int32 arrays
    sell_ticks: Union[NDArray[NDArray[np.int32]], NumbaList[NDArray[np.int32]]]       # Array of 1D int32 arrays
    buy_imbalances: Union[NDArray[NDArray[np.bool_]], NumbaList[NDArray[np.bool_]]]   # Array of 1D bool arrays
    sell_imbalances: Union[NDArray[NDArray[np.bool_]], NumbaList[NDArray[np.bool_]]]  # Array of 1D bool arrays

    # Additional attributes
    cot_price_levels: Optional[NDArray[np.int32]] = None      # 1D int32 array
    sell_imbalances_sum: Optional[NDArray[np.uint16]] = None  # 1D uint16 array
    buy_imbalances_sum: Optional[NDArray[np.uint16]] = None   # 1D uint16 array

    # Private attributes
    _datetime_index: pd.Series = None  # DatetimeIndex for date time slicing (will be set in __post_init__)

    def __post_init__(self):
        # Convert bar_timestamps to pandas DatetimeIndex for easier slicing
        # This method is automatically called after the object is initialized.
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

        Parameters
        ----------
        key : slice or int
            The slice or integer index to access specific data.

        Returns
        -------
        FootprintData
            A new FootprintData object with the sliced data.

        Raises
        ------
        TypeError
            If the key is not a slice or integer.
        """
        if isinstance(key, (slice, int)):
            if isinstance(key, slice) and isinstance(key.start, (str, dt.datetime)) and isinstance(key.stop, (str, dt.datetime)):
                # Use pandas indexing with datetime slice
                start_idx, end_idx = self._datetime_index.slice_locs(start=key.start, end=key.stop)
                return self[start_idx:end_idx]

            # Handle integer index or regular slicing
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
            raise TypeError("Invalid argument type. Expected a slice or integer index.")

    @classmethod
    def from_numba(cls, data: Tuple, price_tick: float) -> 'FootprintData':
        """
        Create a FootprintData instance from the output of the `comp_bar_footprint` function.

        Parameters
        ----------
        data : tuple
            Output of the `comp_bar_footprint` function, containing NumbaList of numpy arrays.
        price_tick : float
            The price tick size.

        Returns
        -------
        FootprintData
            A new instance of FootprintData initialized with the given data.

        Raises
        ------
        ValueError
            If the data is invalid or inconsistent.
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
        Create a FootprintData instance from a dictionary.

        Parameters
        ----------
        data : dict
            Dictionary containing the footprint data with keys:
            'bar_timestamps', 'price_levels', 'buy_volumes', 'sell_volumes',
            'buy_ticks', 'sell_ticks', 'buy_imbalances', 'sell_imbalances', 'price_tick'

        Returns
        -------
        FootprintData
            A new instance of FootprintData initialized with the given data.

        Raises
        ------
        ValueError
            If the data is invalid or inconsistent.
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
        Convert the footprint data to a pandas DataFrame.

        Returns
        -------
        DataFrame
            A pandas DataFrame containing the footprint data.
        """
        df = footprint_to_dataframe(
            self.bar_timestamps,
            self.price_levels,
            self.buy_volumes,
            self.sell_volumes,
            self.buy_ticks,
            self.sell_ticks,
            self.buy_imbalances,
            self.sell_imbalances,
            self.price_tick
        )
        return df

    def cast_to_numba_list(self):
        """
        Cast the footprint data in-place to NumbaList for Numba calculations.

        Notes
        -----
        Numba does not support numpy arrays with dtype=object; we have to cast them to NumbaLists.
        """
        self.price_levels = NumbaList(self.price_levels)
        self.buy_volumes = NumbaList(self.buy_volumes)
        self.sell_volumes = NumbaList(self.sell_volumes)
        self.buy_ticks = NumbaList(self.buy_ticks)
        self.sell_ticks = NumbaList(self.sell_ticks)
        self.buy_imbalances = NumbaList(self.buy_imbalances)
        self.sell_imbalances = NumbaList(self.sell_imbalances)

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
        Calculate the total memory usage of the FootprintData object.

        Returns
        -------
        float
            Total memory usage in megabytes (MB).
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
            if isinstance(array, np.ndarray) or isinstance(array, list) or isinstance(array, NumbaList):
                # Use pympler to measure the size of each element
                total_memory += sum(asizeof.asizeof(item) for item in array if item is not None)

        return total_memory / (1024 ** 2)  # Convert to MB

    def is_valid(self) -> bool:
        """
        Check for consistent lengths and types of all attributes.

        Returns
        -------
        bool
            True if the data is valid, False otherwise.
        """
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
        assert 'timestamp' in trades.columns, "Missing 'timestamp' column in trades data!"
        assert 'price' in trades.columns, "Missing 'price' column in trades data!"
        assert 'amount' in trades.columns, "Missing 'amount' column in trades data!"
        # TODO: support trade side information
        # TODO: Handle Trade splitting on same price level

        self._raw_data = trades
        self._raw_data.sort_values('timestamp', inplace=True)

        self._open_ts = self._open_indices = None
        self._highs = self._lows = None
    
    def __str__(self):
        return (f"Class: {self.__class__.__name__} with members:\n"
                f"{[f"{key}: {value}\n" for key, value in self.__dict__.items()]} "
                f"\nRaw trades data:\n{self._raw_data.info()}")

    @abstractmethod
    def _generate_bar_opens(self) -> Tuple[NDArray[np.int64], NDArray[np.int64]]:
        """
        Abstract method to generate bar indices using the appropriate bar indexer.
        Returns
        -------
        tuple(np.array(np.int64), np.array(np.int64))
            Open timestamps and corresponding open indices in the raw trades data.
        """
        pass

    def _calc_bar_open_values(self):
        """
        Calculate and sets the open timestamps and indices if not already calculated.
        """
        if self._open_indices is None:
            logger.info("Calculating bar open tick indices and timestamps...")
            self._open_ts, self._open_indices = self._generate_bar_opens()

    def build_ohlcv(self) -> pd.DataFrame:
        """
        Build the bar features using the generated indices and raw trades data.
        Returns
        -------
        pd.DataFrame
            A dataframe containing the OHLCV + VWAP features with datetime index corresponding to the bar open timestamps.
        """
        self._calc_bar_open_values()

        ohlcv_tuple = comp_bar_ohlcv(
            self._raw_data['price'].values,
            self._raw_data['amount'].values,
            self._open_indices
        )
        self._highs, self._lows = ohlcv_tuple[1], ohlcv_tuple[2]
        logger.info("OHLCV bars calculated successfully.")

        ohlcv_df = pd.DataFrame({
            'timestamp': self._open_ts[:-1],
            'open': ohlcv_tuple[0],
            'high': ohlcv_tuple[1],
            'low': ohlcv_tuple[2],
            'close': ohlcv_tuple[3],
            'volume': ohlcv_tuple[4],
            'vwap': ohlcv_tuple[5]
        })
        logger.info("OHLCV bars converted to DataFrame.")

        # Convert timestamps to datetime index
        ohlcv_df['timestamp'] = pd.to_datetime(ohlcv_df['timestamp'], unit='ns')
        ohlcv_df.set_index('timestamp', inplace=True)

        return ohlcv_df


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
        self._calc_bar_open_values()

        directional_tuple = comp_bar_directional_features(
            self._raw_data['price'].values,
            self._raw_data['amount'].values,
            self._open_indices
        )
        logger.info("Directional features calculated successfully.")

        directional_df = pd.DataFrame({
            'timestamp': self._open_ts[:-1],
            'ticks_buy': directional_tuple[0],
            'ticks_sell': directional_tuple[1],
            'volume_buy': directional_tuple[2],
            'volume_sell': directional_tuple[3],
            'dollars_buy': directional_tuple[4],
            'dollars_sell': directional_tuple[5],
            'max_spread': directional_tuple[6],
            'cum_volumes_min': directional_tuple[7],
            'cum_volumes_max': directional_tuple[8],
            'cum_dollars_min': directional_tuple[9],
            'cum_dollars_max': directional_tuple[10]
        })
        logger.info("Directional features converted to DataFrame.")

        # Convert timestamps to datetime index
        directional_df['timestamp'] = pd.to_datetime(directional_df['timestamp'], unit='ns')
        directional_df.set_index('timestamp', inplace=True)

        return directional_df

    def build_footprints(self, price_tick_size=None, imbalance_factor=3.0) -> FootprintData:
        """
        Build the footprint data using the generated indices and raw trades data.
        Returns
        -------
        FootprintData
            A FootprintData object containing the footprint data.
        """

        self._calc_bar_open_values()
        if self._highs is None or self._lows is None:
            # We need the bar highs and lows for the footprint calculation
            self.build_ohlcv()

        if price_tick_size is None:
            # Anticipate price tick size
            price_tick_size = comp_price_tick_size(self._raw_data['price'].values)
        logger.info(f"Price tick size: {price_tick_size}")

        # Compute the footprint data
        footprint_data = comp_bar_footprints(
            self._raw_data['price'].values,
            self._raw_data['amount'].values,
            self._open_indices,
            self._open_ts,
            price_tick_size,
            self._lows,
            self._highs,
            imbalance_factor
        )
        logger.info("Footprint data calculated successfully.")

        # Create a FootprintData object
        footprint = FootprintData.from_numba(footprint_data, price_tick_size)
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


@njit(nogil=True, parallel=True)
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


@njit(nogil=True, parallel=False)
def comp_bar_footprints(
    prices: NDArray[np.float64],
    amounts: NDArray[np.float64],
    bar_open_indices: NDArray[np.int64],
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
    NDArray[np.uint16], NDArray[np.uint16], NDArray[np.int32]
]:
    """
    Compute the footprint data and features for each bar.

    Parameters
    ----------
    prices : np.array(np.float64)
        prices of raw trades data
    amounts : np.array(np.float64)
        amounts of raw trades data
    bar_open_indices : np.array(np.int64)
        bar open indices in the raw trades timestamps
    bar_open_timestamps : np.array(np.int64)
        bar open timestamps in nanoseconds
    price_tick_size : float
        price tick size
    bar_lows : np.array(np.float64)
        lows of the bars
    bar_highs : np.array(np.float64)
        highs of the bars
    imbalance_factor : float
        the multiplier factor for the imbalance calculation

    Returns
    -------
    tuple
        - open_timestamps : np.ndarray
        - price_levels : NumbaList[np.ndarray]
        - buy_volumes : NumbaList[np.ndarray]
        - sell_volumes : NumbaList[np.ndarray]
        - buy_ticks : NumbaList[np.ndarray]
        - sell_ticks : NumbaList[np.ndarray]
        - buy_imbalances : NumbaList[np.ndarray]
        - sell_imbalances : NumbaList[np.ndarray]
        - buy_imbalances_sum : np.ndarray
        - sell_imbalances_sum : np.ndarray
        - cot_price_levels : np.ndarray
    Notes
    -----
    The price levels are calculated in (integer) price tick units to eliminate floating point errors.

    """
    # TODO: [IDEA] New data structure; Preallocate Flat Arrays and Indices -> This enables parallelization

    n_bars = len(bar_open_indices) - 1

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

    tick_direction = 0
    for i in prange(n_bars):
        start = bar_open_indices[i]
        end = bar_open_indices[i + 1]

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

        # Start aggregating the footprint data
        for j in range(start, end):
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

        # Calculate the footprint features: buy imbalances, sell imbalances, COT price level
        buy_imbalances_i, sell_imbalances_i, cot_price_level = comp_footprint_features(
            price_levels_i, buy_volumes_i, sell_volumes_i, imbalance_factor
        )
        buy_imbalances.append(buy_imbalances_i)
        sell_imbalances.append(sell_imbalances_i)

        # Update cumulative imbalances and COT price level
        buy_imbalances_sum[i] = np.sum(buy_imbalances_i, dtype=np.uint16)
        sell_imbalances_sum[i] = np.sum(sell_imbalances_i, dtype=np.uint16)
        cot_price_levels[i] = cot_price_level

    return (
        bar_open_timestamps[:n_bars],
        price_levels,
        buy_volumes, sell_volumes,
        buy_ticks, sell_ticks,
        buy_imbalances, sell_imbalances,
        buy_imbalances_sum, sell_imbalances_sum, cot_price_levels
    )


@njit(nogil=True)
def comp_footprint_features(price_levels, buy_volumes, sell_volumes, imbalance_multiplier):
    """
    Calculate bar's footprint features: COT price level, buy imbalances, sell imbalances.

    Parameters
    ----------
    price_levels : np.ndarray
        Price levels of a bar (1D numpy array).
    buy_volumes : np.ndarray
        Buy volumes of a bar (1D numpy array).
    sell_volumes : np.ndarray
        Sell volumes of a bar (1D numpy array).
    imbalance_multiplier : float
        Imbalance multiplier.

    Returns
    -------
    buy_imbalances : np.ndarray
        1D boolean array of buy imbalances.
    sell_imbalances : np.ndarray
        1D boolean array of sell imbalances.
    cot_price_level : int
        Commitment of Traders (COT) price level.
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
        sell_imbalances[:-1] = sell_volumes[:-1] > (buy_volumes[1:] * imbalance_multiplier)
        buy_imbalances[1:] = buy_volumes[1:] > (sell_volumes[:-1] * imbalance_multiplier)

    sum_level_volume = buy_volumes + sell_volumes
    highest_cot_idx = np.argmax(sum_level_volume)
    cot_price_level = price_levels[highest_cot_idx]

    return buy_imbalances, sell_imbalances, cot_price_level
