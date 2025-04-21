import datetime as dt
from dataclasses import dataclass
from typing import Union, Optional, Tuple, Dict
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from numba.typed import List as NumbaList

from finmlkit.bar.utils import footprint_to_dataframe


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
        Timestamps of the bar.
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
