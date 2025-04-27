import datetime as dt
from dataclasses import dataclass
from typing import Union, Optional, Tuple, Dict
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from numba.typed import List as NumbaList

from finmlkit.bar.utils import footprint_to_dataframe
from finmlkit.utils.log import get_logger
from .utils import comp_trade_side_vector

logger = get_logger(__name__)


class TradesData:
    """
    Class to preprocess trades data for bar building.

    This class handles standardization of column names, timestamp conversion,
    trade merging, and side inference for consistent processing across different
    data sources.
    """

    def __init__(self,
                 trades: pd.DataFrame,
                 columns: Optional[dict[str, str]] = None,
                 timestamp_unit: Optional[str] = None,
                 proc_res: Optional[str] = None,
                 inplace: bool = False):
        """
        Initialize the TradesData with raw trades data.

        :param trades: DataFrame containing raw trades data with 'timestamp', 'price',
                      and 'amount' columns. If 'is_buyer_maker' is present, it indicates
                      the trade side; otherwise, it is inferred.
        :param columns: (Optional) Dictionary to map raw column names to standard names: `timestamp`, `price`, `amount`, `is_buyer_maker` (optional).
                        Eg.: {'time': 'timestamp', 'qty': 'amount', 'is_buyer_maker': 'side'}
        :param timestamp_unit: (Optional) timestamp unit (e.g., 'ms', 'us', 'ns'); inferred if None.
        :param proc_res: (Optional) processing resolution for timestamps (e.g., 'ms' cuts us to ms resolution).
        :param inplace: If True, modifies the trades DataFrame in place.
        :raises ValueError: If required columns are missing or timestamp format is invalid.

        """
        if not inplace:
            trades = trades.copy()

        # Process the trades data step by step
        self._preprocess_column_names(trades, columns)
        self._validate_columns(trades)
        self.is_side = self._is_side_column_present(trades)
        self._sort_trades(trades)
        trades = self._merge_trades(trades)
        self._timestamp_unit = self._convert_timestamps_to_ns(trades, timestamp_unit)
        self._apply_timestamp_resolution(trades, timestamp_unit, proc_res)
        self._extract_trade_side_info(trades)

        # Store the processed data
        self.data = trades

        logger.info("TradesData prepared successfully.")

    @staticmethod
    def _preprocess_column_names(trades: pd.DataFrame, columns: dict[str, str]) -> None:
        """
        Standardize column names for consistent processing.

        :param trades: DataFrame to preprocess.
        """
        if columns:
            standard_column_names = ['timestamp', 'price', 'amount', 'is_buyer_maker']
            # Ensure the provided columns are valid
            for key, value in columns.items():
                if key not in trades.columns:
                    raise ValueError(f"Missing required column: {key}")
                if value not in standard_column_names:
                    raise ValueError(f"Invalid column mapping: {value}. Must be one of: {', '.join(standard_column_names)}")
                trades.rename(columns={key: value}, inplace=True)

    @staticmethod
    def _is_side_column_present(trades: pd.DataFrame) -> bool:
        return "is_buyer_maker" in trades.columns

    @staticmethod
    def _validate_columns(trades: pd.DataFrame) -> None:
        """
        Validate that required columns are present in the DataFrame.

        :param trades: DataFrame to validate.
        :raises ValueError: If required columns are missing.
        """
        required_columns = ['timestamp', 'price', 'amount']
        missing_columns = [col for col in required_columns if col not in trades.columns]

        if missing_columns:
            raise ValueError(f"Missing columns in trades data: {', '.join(missing_columns)}")

    @staticmethod
    def _sort_trades(trades: pd.DataFrame) -> None:
        """
        Sort trades by timestamp to ensure correct order for processing.

        :param trades: DataFrame to sort.
        """
        logger.info('Input trades data OK. Sorting by timestamp...')
        trades.sort_values(by='timestamp', inplace=True)
        trades.reset_index(drop=True, inplace=True)

    def _merge_trades(self, trades: pd.DataFrame) -> pd.DataFrame:
        """
        Merge trades that occur at the same timestamp and price level.

        :param trades: DataFrame with trades to merge.
        :return: Merged DataFrame.
        """
        # TODO: Implement a more efficient numba merging strategy if needed
        logger.info('Merging split trades (same timestamps) on same price level...')
        if self.is_side:
            trades = trades.groupby(['timestamp', 'price', 'is_buyer_maker'],
                                           as_index=False).agg({'amount': 'sum'})
        else:
            trades = trades.groupby(['timestamp', 'price'],
                                           as_index=False).agg({'amount': 'sum'})

        return trades

    def _convert_timestamps_to_ns(self, trades: pd.DataFrame, timestamp_unit: Optional[str]) -> str:
        """
        Convert timestamps to nanosecond representation.

        :param trades: DataFrame with timestamps to convert.
        :param timestamp_unit: Unit of the timestamp values or None to infer.
        :return: The timestamp unit used for conversion.
        :raises ValueError: If timestamp format is invalid.
        """
        # Infer or validate timestamp unit
        if timestamp_unit is None:
            timestamp_unit = self.infer_ts_unit(trades)
        valid_units = ['s', 'ms', 'us', 'ns']
        if timestamp_unit not in valid_units:
            raise ValueError(f"Invalid timestamp format! Must be one of: {', '.join(valid_units)}")

        # Convert timestamp to nanoseconds
        logger.info('Converting timestamp to nanoseconds units for processing...')
        trades.timestamp = pd.to_datetime(trades.timestamp, unit=timestamp_unit).astype(np.int64).values

        return timestamp_unit

    @staticmethod
    def _apply_timestamp_resolution(trades: pd.DataFrame, timestamp_unit: str, proc_res: Optional[str]) -> None:
        """
        Apply processing resolution to timestamps if specified.

        :param trades: DataFrame with timestamps already converted to nanoseconds.
        :param timestamp_unit: Current timestamp unit (should be 'ns' at this point).
        :param proc_res: Target processing resolution for timestamps.
        :raises ValueError: If processing resolution is invalid.
        """
        if proc_res and proc_res != timestamp_unit:
            logger.info(f"Processing resolution: {proc_res} -> converting timestamps...")

            # Convert proc_res to nanoseconds scale factor
            scale_factors = {'s': 1_000_000_000, 'ms': 1_000_000, 'us': 1_000, 'ns': 1}
            if proc_res not in scale_factors:
                raise ValueError(
                    f"Invalid processing resolution: {proc_res}. Must be one of: {', '.join(scale_factors.keys())}")

            # Round timestamps to the specified resolution
            resolution_ns = scale_factors[proc_res]
            trades.timestamp = (trades.timestamp // resolution_ns) * resolution_ns

    def _extract_trade_side_info(self, trades: pd.DataFrame) -> None:
        """
        Extract trade side information from the trades data.

        :param trades: DataFrame to process.
        :returns: None - modifies the trades DataFrame in place to include a 'side' column.
        """
        if self.is_side:
            logger.info("Trade side information found. Using 'is_buyer_maker' to determine trade side.")
            trades['side'] = np.where(trades['is_buyer_maker'] == 1, -1, 1).astype(np.int8)
        else:
            logger.info("No trade side information found. Inferring trade side from price movements.")
            trades['side'] = comp_trade_side_vector(trades['price'].values)

    @staticmethod
    def infer_ts_unit(trades: pd.DataFrame) -> str:
        """
        Infer the unit of timestamps in the trades data if not explicitly provided.
        :param trades: DataFrame containing timestamp data.
        :return: Inferred or provided timestamp unit.
        """
        if trades.empty:
            logger.warning("Empty trades DataFrame, defaulting to millisecond timestamps.")
            return 'ms'

        max_ts = trades['timestamp'].max()

        if max_ts > 1e18:  # Likely in nanoseconds
            timestamp_unit = 'ns'
        elif max_ts > 1e15:  # Likely in microseconds
            timestamp_unit = 'us'
        elif max_ts > 1e12:  # Likely in milliseconds
            timestamp_unit = 'ms'
        else:  # Likely in seconds
            timestamp_unit = 's'
            logger.warning("Timestamp unit is set to seconds. Please verify the data.")

        logger.info(f"Inferred timestamp format: {timestamp_unit}")

        return timestamp_unit

    @property
    def timestamp_unit(self) -> str:
        """
        Get the timestamp unit used for processing.

        :return: Timestamp unit string.
        """
        return self._timestamp_unit


@dataclass
class FootprintData:
    """
    Container for dynamic memory footprint calculations including trade volumes, price levels, and imbalance information.

    :param bar_timestamps: Timestamps of each bar in nanoseconds.
    :param price_tick: Price tick size.
    :param price_levels: Array of price levels per bar.
    :param buy_volumes: Buy volumes per price level.
    :param sell_volumes: Sell volumes per price level.
    :param buy_ticks: Number of buy ticks per price level.
    :param sell_ticks: Number of sell ticks per price level.
    :param buy_imbalances: Buy imbalance flags per price level.
    :param sell_imbalances: Sell imbalance flags per price level.
    :param cot_price_levels: Optional Commitment of Traders price levels.
    :param sell_imbalances_sum: Optional total sell imbalance counts per bar.
    :param buy_imbalances_sum: Optional total buy imbalance counts per bar.
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
        Return the number of bars in the data.
        :returns: Number of bars.
        """
        return len(self.bar_timestamps)

    def __repr__(self) -> str:
        """
        Generate a summary string representation for debugging.
        :returns: Formatted string summary.
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
        Support slicing or indexing of the footprint data.
        :param key: Slice, integer index, or datetime range.
        :returns: New FootprintData object with selected range.
        :raises TypeError: If key is not a supported type.
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
        Create a FootprintData object from Numba-based output.
        :param data: Output tuple from comp_bar_footprint.
        :param price_tick: Tick size for price levels.
        :returns: A validated FootprintData instance.
        :raises ValueError: If data length is inconsistent.
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
        Create a FootprintData object from a dictionary of arrays.
        :param data: Dictionary with raw footprint arrays.
        :returns: A validated FootprintData instance.
        :raises ValueError: If data length is inconsistent.
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
        Convert the footprint data into a pandas DataFrame.
        :returns: A DataFrame with structured footprint information.
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
        Convert internal arrays to NumbaList for JIT-compatible processing.
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
        Convert internal lists to NumPy arrays for general-purpose processing.
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
        Estimate memory usage of all internal arrays.
        :returns: Memory usage in megabytes (MB).
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
        Check if all internal arrays are consistent.
        :returns: True if valid, False otherwise.
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
