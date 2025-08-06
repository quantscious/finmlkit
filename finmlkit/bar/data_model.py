import datetime as dt
from dataclasses import dataclass
from typing import Union, Optional, Tuple, Dict
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from numba.typed import List as NumbaList
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import sys

from finmlkit.bar.utils import footprint_to_dataframe
from finmlkit.utils.log import get_logger
from .utils import comp_trade_side_vector, merge_split_trades
import os

logger = get_logger(__name__)


class TradesData:
    r"""Preprocessor class for raw trades data, designed for efficient bar building and financial analysis.

    This class handles standardization of column names, timestamp conversion, trade merging, side inference,
    and data validation for consistent processing across different data sources. It serves as the primary
    data preparation component for high-frequency trading analysis and bar construction workflows.

    In high-frequency trading data, raw trades often come in various formats with inconsistent timestamps,
    split trades at the same price level, missing side information, and data integrity issues. This class
    addresses these challenges by providing a robust preprocessing pipeline that:

    - **Normalizes timestamps** to nanosecond precision for consistent temporal analysis
    - **Merges split trades** that occur at identical timestamps and price levels to reduce noise
    - **Infers trade sides** (buyer/seller initiated) when not explicitly provided
    - **Validates data integrity** by detecting trade ID discontinuities and temporal inconsistencies
    - **Provides efficient storage** via compressed HDF5 format with monthly partitioning
    - **Enables time-slice queries** with multiprocessing support for large datasets

    The preprocessing pipeline follows these steps when ``preprocess=True``:

    1. **Timestamp Conversion**: Convert to nanosecond precision from various units (s, ms, μs, ns)
    2. **Data Sorting**: Sort by trade ID first to detect gaps, then by timestamp for chronological order
    3. **Trade Merging**: Aggregate trades with identical timestamps and prices.

    4. **Resolution Processing**: Optionally round timestamps to specified resolution (e.g., millisecond)
    5. **Side Inference**: Determine trade direction from price movements when side data is unavailable

    The class supports **monthly HDF5 partitioning** for efficient storage and retrieval of large datasets.
    Each month's data is stored under ``/trades/YYYY-MM`` with accompanying metadata for fast range queries.
    This approach enables handling multi-terabyte datasets while maintaining query performance.

    **Data Integrity Monitoring**: The class tracks discontinuities in trade IDs and timestamps, computing
    missing data percentages and flagging potential data quality issues. This is crucial for ensuring
    reliable downstream analysis.

    .. tip::
        For optimal performance with large datasets (>10Gb trades), enable preprocessing and use HDF5 storage
        with compression. The class automatically handles memory-efficient processing via chunking and
        can leverage multiprocessing for data loading operations.

    .. note::
        Trade side inference uses price tick rule and other heuristics when explicit side information
        is unavailable. For critical applications, prefer data sources with explicit buyer/seller flags.

    Args:
        ts (NDArray): Array of timestamps in various units (s, ms, μs, ns). Must be numeric and monotonic.
        px (NDArray): Array of trade prices as floating-point values.
        qty (NDArray): Array of trade quantities/amounts as floating-point values.
        id (NDArray, optional): Array of unique trade identifiers for data validation. Required if ``preprocess=True``.
        is_buyer_maker (NDArray, optional): Boolean array indicating buyer-maker status (True if buyer is maker).
            If provided, used for accurate side determination.
        side (NDArray, optional): Pre-computed trade side array (-1: sell, 1: buy). Used when loading from HDF5.
        dt_index (pd.DatetimeIndex, optional): Pre-computed datetime index. If None, created from timestamps.
        timestamp_unit (str, optional): Explicit timestamp unit ('s', 'ms', 'us', 'ns'). Auto-inferred if None.
        preprocess (bool, optional): Enable full preprocessing pipeline. Default: False.
        proc_res (str, optional): Target timestamp resolution for rounding ('ms', 'us'). Default: None (no rounding).
        name (str, optional): Instance name for logging purposes. Default: None.

    Raises:
        TypeError: If input arrays are not numpy ndarrays or have incompatible types.
        ValueError: If required columns are missing, timestamp format is invalid, or preprocessing fails.

    Examples:
        Basic usage with preprocessing:

        >>> import numpy as np
        >>> import pandas as pd
        >>> from finmlkit.bar.data_model import TradesData
        >>> # Raw trades data
        >>> timestamps = np.array([1609459200000, 1609459201000, 1609459202000])  # ms
        >>> prices = np.array([100.0, 100.5, 99.8])
        >>> quantities = np.array([1.5, 2.0, 0.8])
        >>> trade_ids = np.array([1001, 1002, 1003])
        >>>
        >>> # Create TradesData with preprocessing
        >>> trades = TradesData(timestamps, prices, quantities, trade_ids,
        ...                     timestamp_unit='ms', preprocess=True, name='BTCUSD')
        >>> print(f"Processed {len(trades.data)} trades")
        Processed 3 trades

        Loading from HDF5 with time filtering:

        >>> # doctest: +SKIP
        >>> # Load specific time range with multiprocessing
        >>> trades = TradesData.load_trades_h5('trades.h5',
        ...                                     start_time='2021-01-01',
        ...                                     end_time='2021-01-31',
        ...                                     enable_multiprocessing=True)
        >>> trades.set_view_range('2021-01-15', '2021-01-20')
        >>> subset = trades.data  # Only data in view range

    See Also:
        :class:`finmlkit.bar.base.BarBuilderBase`: Uses TradesData for constructing various bar types.
        :func:`finmlkit.bar.utils.merge_split_trades`: Core function for trade aggregation.
        :func:`finmlkit.bar.utils.comp_trade_side_vector`: Trade side inference algorithm.

    References:
        .. _`HDF5 for High-Frequency Trading`: https://www.hdfgroup.org/
        .. _`Market Microstructure in Practice`: https://www.cambridge.org/core/books/market-microstructure-in-practice/
    """

    def __init__(self,
                 ts: NDArray, px: NDArray, qty: NDArray, id: NDArray = None, *,
                 is_buyer_maker: NDArray = None,
                 side = None,
                 dt_index: Optional[pd.DatetimeIndex] = None,
                 timestamp_unit: Optional[str] = None,
                 preprocess: bool = False,
                 proc_res: Optional[str] = None, name= None):
        """
        Initialize the TradesData with raw trades data.

        :param ts: array of timestamps
        :param px: array of prices
        :param qty: array of quantity or amount of trades
        :param id: array of trades id
        :param is_buyer_maker: Optional Array of side info: True if buyer maker, False otherwise. If None side information will be inferred from data.
        :param side: Optional Array Market order side information (-1: sell, 1: buy) [needed when loading from HDF5 store].
        :param dt_index: Optional DatetimeIndex for the trades data. If provided, it will be used as the index.  [needed when loading from HDF5 store].
        :param timestamp_unit: (Optional) timestamp unit (e.g., 'ms', 'us', 'ns'); inferred if None.
        :param proc_res: (Optional) processing resolution for timestamps (e.g., 'ms' cuts us to ms resolution).
        :param preprocess: If True, runs the preprocessing pipeline (sorting, merging split trades etc...)
        :param name: Optional name for the trades data instance (logging purposes).
        :raises ValueError: If required columns are missing or timestamp format is invalid.

        """
        if not isinstance(ts, np.ndarray):
            raise TypeError("ts must be a np.ndarray")
        if not isinstance(px, np.ndarray):
            raise TypeError("px must be a np.ndarray")
        if not isinstance(qty, np.ndarray):
            raise TypeError("qty must be a np.ndarray")
        if id is not None and not isinstance(id, np.ndarray):
            raise TypeError("id must be a np.ndarray")
        if is_buyer_maker is not None and not isinstance(is_buyer_maker, np.ndarray):
            raise TypeError("is_buyer_maker must be None or np.ndarray")
        if side is not None and not isinstance(side, np.ndarray):
            raise TypeError("side must be None or np.ndarray")

        self._start_date = self._end_date = None


        self._data = pd.DataFrame({'timestamp': ts, 'price': px, 'amount': qty, 'id': id})

        self.is_buyer_maker = is_buyer_maker
        if side is not None:
            self._data['side'] = side
        self._orig_timestamp_unit = timestamp_unit if timestamp_unit else self._infer_timestamp_unit()
        self.name = name

        # Process the trades data
        self.missing_pct = 0
        self.data_ok = None
        self.discontinuities = []  # List to store discontinuity information
        if preprocess:
            if id is None:
                raise ValueError("id is required if preprocess is True")
            self._convert_timestamps_to_ns()
            self._sort_trades()
            self._merge_trades()
            self._apply_timestamp_resolution(proc_res)
            if "side" not in self._data.columns:
                # If side info is not provided, infer it from trades data
                self._add_trade_side_info()

        # Add datetime_idx
        if dt_index is not None:
            self._data.set_index(dt_index, inplace=True)
        else:
            self._data.set_index(pd.to_datetime(self._data['timestamp'], unit='ns'), inplace=True)
            self._data.index.name = "datetime"
            logger.info("TradesData prepared successfully.")


    @property
    def start_date(self):
        """
        Get the start date of the trades data.

        :return: Start date as a pandas Timestamp.
        """
        return self._start_date

    @property
    def end_date(self):
        """
        Get the end date of the trades data.

        :return: End date as a pandas Timestamp.
        """
        return self._end_date

    def set_view_range(self, start: pd.Timestamp|str, end: pd.Timestamp|str):
        r"""Set the active view range for data access, enabling efficient time-slice analysis.

        :param start: Start timestamp for the view range. Accepts string or pd.Timestamp.
        :param end: End timestamp for the view range. Accepts string or pd.Timestamp.
        :raises ValueError: If start timestamp is not before end pd.timestamp.
        """
        if isinstance(start, str):
            start = pd.Timestamp(start)
        if isinstance(end, str):
            end = pd.Timestamp(end)
        if start >= end:
            raise ValueError("Start timestamp must be before end timestamp.")

        self._start_date = start
        self._end_date = end

        logger.info(f"View range set to {start} - {end}.")

    @property
    def data(self) -> pd.DataFrame:
        r"""Get the processed trades data as a DataFrame, respecting the active view range.

        Returns the full dataset if no view range is set, or a time-filtered subset otherwise.
        The DataFrame includes columns: timestamp, price, amount, and optionally side.

        :return: DataFrame containing trades data with datetime index.
        """
        if self._start_date is None and self._end_date is None:
            return self._data

        return self._data.loc[self._start_date: self._end_date]

    @property
    def orig_timestamp_unit(self) -> str:
        """
        Get the timestamp unit used for processing.

        :return: Timestamp unit string.
        """
        return self._orig_timestamp_unit

    def _validate_data(self):
        """
        Check for gaps in trade IDs
        """
        # First convert to numeric to handle potential string IDs
        id_diffs = np.diff(self.data['id'].values)
        gap_indices = np.where(id_diffs > 1)[0]
        cum_gap_size = 0
        if len(gap_indices) > 0:
            logger.warning(f"{self.name} | Found {len(gap_indices):,} discontinuities in trade IDs. "
                           f"This indicates missing trades.")
            # Record detailed information about each discontinuity
            n_large_gaps = 0
            for idx in gap_indices:
                gap_start_id = int(self.data['id'].iloc[idx])
                gap_end_id = int(self.data['id'].iloc[idx + 1])
                gap_size = gap_end_id - gap_start_id - 1
                cum_gap_size += gap_size

                # Get timestamps for the trades before and after the gap
                pre_gap_time = pd.to_datetime(self.data['timestamp'].iloc[idx], unit='ns')
                post_gap_time = pd.to_datetime(self.data['timestamp'].iloc[idx + 1], unit='ns')
                time_diff = post_gap_time - pre_gap_time

                # Record the discontinuity if gap is greater than 1 min
                if time_diff > pd.Timedelta(minutes=1):
                    self.data_ok = False
                    n_large_gaps += 1
                    self.discontinuities.append({
                        'start_id': gap_start_id,
                        'end_id': gap_end_id,
                        'missing_ids': gap_size,
                        'pre_gap_time': pre_gap_time,
                        'post_gap_time': post_gap_time,
                        'time_interval': time_diff
                    })
            if n_large_gaps > 0:
                logger.warning(f"{self.name} | Found {n_large_gaps} large gaps greater than 1 minute.")
            self.missing_pct = cum_gap_size / len(self.data) * 100

    def _sort_trades(self) -> None:
        """
        Sort trades by timestamp to ensure correct order for processing.
        Also performs data integrity checks by identifying discontinuities in trade IDs.
        """
        self.data_ok = True
        self.discontinuities = []  # Reset discontinuities list

        # Sort by ID to inspect data integrity
        self.data.sort_values(by=['id'], inplace=True)
        # Reset index
        self.data.reset_index(drop=True, inplace=True)

        # Check duplicates in trade IDs
        if self.data['id'].duplicated().any():
            logger.warning(f"{self.name} | Trade IDs contain duplicates. This may indicate data corruption.")
            # Drop duplicates while keeping the first occurrence
            self.data.drop_duplicates(subset='id', keep='first', inplace=True)
            logger.info("Duplicates in trade IDs have been removed.")
            self.data_ok = False

        self._validate_data()

        # Now sort by timestamp for chronological order if needed
        if not self.data.timestamp.is_monotonic_increasing:
            logger.warning(f"{self.name} | Trades timestamps are not monotonic increasing after sorting by trade IDs. "
                         f"Sorting by timestamp for chronological order...")
            self.data.sort_values(by=['timestamp', 'id'], inplace=True)

        # Reset index
        self.data.reset_index(drop=True, inplace=True)

    def _merge_trades(self):
        """
        Merge trades that occur at the same timestamp and price level.
        """
        logger.info('Merging split trades (same timestamps) on same price level...')

        ts, px, am, side = merge_split_trades(
            self.data['timestamp'].values.astype(np.int64),
            self.data['price'].values.astype(np.float64),
            self.data['amount'].values.astype(np.float32),
            self.is_buyer_maker,
        )
        self._data = pd.DataFrame({
            'timestamp': ts,
            'price': px,
            'amount': am
        })
        if self.is_buyer_maker is not None:
            self._data['side'] = side

    def _convert_timestamps_to_ns(self):
        """
        Convert timestamps to nanosecond representation.
        :raises ValueError: If timestamp format is invalid.
        """
        # Infer or validate timestamp unit
        valid_units = ['s', 'ms', 'us', 'ns']
        if self.orig_timestamp_unit not in valid_units:
            raise ValueError(f"Invalid timestamp format! Must be one of: {', '.join(valid_units)}")

        unit_scale_factor = {
            's': 1_000_000_000,
            'ms': 1_000_000,
            'us': 1_000,
            'ns': 1
        }

        # Convert timestamp to nanoseconds
        logger.info('Converting timestamp to nanoseconds units for processing...')
        # trades.timestamp = pd.to_datetime(trades.timestamp, unit=timestamp_unit).astype(np.int64).values
        # Work directly on the underlying NumPy array for better performance
        factor = unit_scale_factor[self.orig_timestamp_unit]
        self._data['timestamp'].values[:] = np.multiply(self.data['timestamp'].values, factor, dtype=np.int64)

    def _apply_timestamp_resolution(self, proc_res: Optional[str]) -> None:
        """
        Apply processing resolution to timestamps if specified.

        :param proc_res: Target processing resolution for timestamps.
        :raises ValueError: If processing resolution is invalid.
        """
        if proc_res and proc_res != self.orig_timestamp_unit:
            logger.info(f"Processing resolution: {proc_res} -> converting timestamps...")

            # Convert proc_res to nanoseconds scale factor
            scale_factors = {'s': 1_000_000_000, 'ms': 1_000_000, 'us': 1_000, 'ns': 1}
            if proc_res not in scale_factors:
                raise ValueError(
                    f"Invalid processing resolution: {proc_res}. Must be one of: {', '.join(scale_factors.keys())}")

            # Round timestamps to the specified resolution
            resolution_ns = scale_factors[proc_res]
            self.data.timestamp = (self.data.timestamp // resolution_ns) * resolution_ns

    def _add_trade_side_info(self) -> None:
        """
        Extract trade side information from the trades data.

        :returns: None - modifies the trades DataFrame in place to include a 'side' column.
        """
        logger.info("No trade side information found. Inferring trade side from price movements.")
        self._data['side'] = comp_trade_side_vector(self.data['price'].values)

    def _infer_timestamp_unit(self) -> str:
        """
        Infer the unit of timestamps in the trades data if not explicitly provided.
        :return: Inferred or provided timestamp unit.
        """
        max_ts = self.data['timestamp'].max()

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

    def save_h5(
            self,
            filepath: str,
            *,
            month_key: Optional[str] = None,
            complib: str = "blosc:lz4",
            complevel: int = 1,
            mode: str = "a",
            chunksize: int = 1_000_000,
            overwrite_month: bool = True,
    ) -> str:
        r"""Persist trades data to HDF5 format with monthly partitioning and compression.

        Stores data under ``/trades/YYYY-MM`` groups for efficient range queries. Each month
        includes metadata for fast discovery and data integrity information when available.

        :param filepath: Destination HDF5 file path. Parent directories created automatically.
        :param month_key: Override automatic monthly key derivation (format: "YYYY-MM").
        :param complib: Compression library ("blosc:lz4", "blosc:zstd", "zlib"). Default: "blosc:lz4".
        :param complevel: Compression level (0-9). Higher values increase compression ratio. Default: 1.
        :param mode: File access mode ("a" for append, "w" for overwrite). Default: "a".
        :param chunksize: Row chunk size for writing large datasets. Default: 1,000,000.
        :param overwrite_month: Prompt for confirmation when overwriting existing monthly data. Default: True.
        :returns: Full HDF5 key path used for storage (e.g., "/trades/2021-03").
        :raises ValueError: If user declines to overwrite existing data or if data format is invalid.
        """
        # ------------------------------------------------------------------
        #  Derive the monthly key and ensure output path exists
        # ------------------------------------------------------------------
        if month_key is None:
            first_dt = pd.to_datetime(self.data["timestamp"].iloc[0], unit="ns")
            month_key = f"{first_dt.year:04d}-{first_dt.month:02d}"

        h5_key = f"/trades/{month_key}"
        meta_key = f"/meta/{month_key}"
        integrity_key = f"/integrity/{month_key}"

        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)

        # ------------------------------------------------------------------
        #  Build an *indexed* frame for fast time‑slice queries
        # ------------------------------------------------------------------
        # frame = self.data.copy()
        # frame["datetime"] = pd.to_datetime(frame["timestamp"], unit="ns")
        # frame.set_index("datetime", inplace=True)
        # It is already indexed!
        frame = self.data

        # ------------------------------------------------------------------
        #  Write / append to the store
        # ------------------------------------------------------------------
        with pd.HDFStore(
                filepath,
                mode=mode,
                complib=complib,
                complevel=complevel,
        ) as store:
            # Check if month data already exists
            month_exists = h5_key in store
            should_overwrite = False

            if month_exists and overwrite_month:
                # Prompt user for confirmation
                #record_count = store.get_storer(h5_key).nrows
                #user_input = input(
                #    f"WARNING: Data for {month_key} already exists with {record_count:,} records.\n"
                #    f"Do you want to overwrite it? [y/N]: "
                #).lower()
                #should_overwrite = user_input in ('y', 'yes')

                should_overwrite = True
                if not should_overwrite:
                    user_input = input("Do you want to append to existing data instead? [Y/n]: ").lower()
                    if user_input in ('n', 'no'):
                        logger.info(f"Operation cancelled by user. No changes made to {month_key} data.")
                        return h5_key

            # Handle data writing
            if month_exists and should_overwrite:
                logger.info(f"Overwriting existing data for {month_key}...")
                store.remove(h5_key)
                store.remove(meta_key)
                if integrity_key in store:
                    store.remove(integrity_key)
                store.put(
                    key=h5_key,
                    value=frame,
                    format="table",
                    data_columns=["timestamp"],
                    index=False,
                    min_itemsize={"side": 1},
                )
            elif month_exists:
                # Append using PyTables row‑wise interface (fast)
                logger.info(f"Appending to existing data for {month_key}...")
                store.append(
                    key=h5_key,
                    value=frame,
                    format="table",
                    data_columns=["timestamp"],
                    index=False,
                    min_itemsize={"side": 1},
                    chunksize=chunksize,
                )
            else:
                # Create new month data
                logger.info(f"Creating new data for {month_key}...")
                store.put(
                    key=h5_key,
                    value=frame,
                    format="table",
                    data_columns=["timestamp"],
                    index=False,
                    min_itemsize={"side": 1},
                )

            # ------------------------------------------------------------------
            #  Update lightweight per‑group metadata (fast group discovery later)
            # ------------------------------------------------------------------
            meta = pd.Series(
                {
                    "record_count": len(frame) if should_overwrite else (
                        store.get_storer(h5_key).nrows if month_exists else len(frame)),
                    "first_timestamp": int(frame["timestamp"].iloc[0]),
                    "last_timestamp": int(frame["timestamp"].iloc[-1]),
                    "data_integrity_ok": self.data_ok,  # Add integrity flag to main metadata
                    "missing_pct": self.missing_pct     # Add count of discontinuities
                }
            )
            store.put(meta_key, meta, format="fixed")

            # ------------------------------------------------------------------
            #  Store data integrity information if discontinuities were found
            # ------------------------------------------------------------------
            if self.discontinuities:
                # Convert discontinuities list to a DataFrame with string representation of objects
                discontinuity_data = []
                for disc in self.discontinuities:
                    # Convert pandas Timestamp and Timedelta objects to strings to ensure serialization works
                    disc_dict = {
                        'start_id': disc['start_id'],
                        'end_id': disc['end_id'],
                        'missing_ids': disc['missing_ids'],
                        'pre_gap_time_str': str(disc['pre_gap_time']),
                        'post_gap_time_str': str(disc['post_gap_time']),
                        'time_interval_str': str(disc['time_interval'])
                    }
                    discontinuity_data.append(disc_dict)

                # Save discontinuity data as DataFrame
                if discontinuity_data:
                    disc_df = pd.DataFrame(discontinuity_data)
                    store.put(integrity_key, disc_df, format="table")
                    logger.info(f"Saved {len(disc_df)} trade ID discontinuities to metadata.")

            logger.info(f"Successfully saved {len(frame):,} records for {month_key}")

        return h5_key

    # ------------------------------------------------------------------
    #  Reading helpers
    # ------------------------------------------------------------------
    @classmethod
    def _keys_for_timerange(
        cls, store: pd.HDFStore, start: Optional[pd.Timestamp], end: Optional[pd.Timestamp]
    ) -> list[str]:
        """Internal helper – determine which monthly groups intersect the
        *[start, end]* interval by consulting the per‑group metadata.
        """
        candidate_keys: list[str] = []
        for meta_key in (k for k in store.keys() if k.startswith("/meta/")):
            meta = store[meta_key]
            first = pd.to_datetime(meta["first_timestamp"], unit="ns")
            last = pd.to_datetime(meta["last_timestamp"], unit="ns")
            if (end is None or first <= end) and (start is None or last >= start):
                # If there is intersection, add the corresponding trades key
                candidate_keys.append(meta_key.replace("/meta", "/trades"))
        return sorted(candidate_keys)

    @classmethod
    def load_trades_h5(
        cls,
        filepath: str,
        *,
        key: Optional[str] = None,
        start_time: Optional[Union[str, pd.Timestamp]] = None,
        end_time: Optional[Union[str, pd.Timestamp]] = None,
        n_workers: Optional[int] = None,
        enable_multiprocessing: bool = True,
        min_groups_for_mp: int = 2,
    ) -> "TradesData":
        r"""Load trades from HDF5 storage with optional multiprocessing and time filtering.

        Supports three loading modes:

            1. **Single month**: Load specific monthly partition using ``key`` parameter
            2. **Time range**: Auto-discover monthly groups intersecting ``[start_time, end_time]``
            3. **Filtered month**: Combine ``key`` with time range for constrained loading

        Multiprocessing is automatically enabled for loading multiple monthly groups,
        significantly improving performance for large time ranges.

        :param filepath: Path to HDF5 file containing trades data.
        :param key: Specific monthly key to load (e.g., "2021-03"). If None, uses time range discovery.
        :param start_time: Start time for filtering (string or Timestamp). None for no start limit.
        :param end_time: End time for filtering (string or Timestamp). None for no end limit.
        :param n_workers: Number of worker processes. If None, uses CPU count - 1.
        :param enable_multiprocessing: Enable parallel loading for multiple groups. Default: True.
        :param min_groups_for_mp: Minimum groups required to trigger multiprocessing. Default: 2.
        :returns: TradesData instance with loaded and concatenated data.
        :raises KeyError: If specified key doesn't exist or no groups match the time range.
        :raises ValueError: If no data is successfully loaded from any group.

        Examples:
            Load specific month:

            >>> # doctest: +SKIP
            >>> trades = TradesData.load_trades_h5('data.h5', key='2021-03')

            Load time range with multiprocessing:

            >>> # doctest: +SKIP
            >>> trades = TradesData.load_trades_h5('data.h5',
            ...                                     start_time='2021-01-01',
            ...                                     end_time='2021-12-31',
            ...                                     n_workers=4)
        """
        # ------------------------------------------------------------------
        #  Normalise temporal boundaries
        # ------------------------------------------------------------------
        if isinstance(start_time, str):
            start_time = pd.Timestamp(start_time)
        if isinstance(end_time, str):
            end_time = pd.Timestamp(end_time)

        logger.info(f"Loading trades from {filepath}...")

        # First, determine which keys we need to load
        with pd.HDFStore(filepath, mode="r") as store:
            available_keys = [k for k in store.keys() if k.startswith('/trades/')]

            # Determine which groups to read -----------------------------------------------------
            if key is not None:
                h5_keys = [f"/trades/{key}"]
                # Check if key is available
                if h5_keys[0] not in store:
                    logger.info(f"Available keys in the store: {available_keys}")
                    raise KeyError(f"HDF5 group '{h5_keys[0]}' not found in the store.")
            else:
                h5_keys = cls._keys_for_timerange(store, start_time, end_time)

            if not h5_keys:
                raise KeyError("No HDF5 group matches the requested slice.")

        # ------------------------------------------------------------------
        #  Decide whether to use multiprocessing
        # ------------------------------------------------------------------
        use_multiprocessing = (
            enable_multiprocessing and
            len(h5_keys) >= min_groups_for_mp
        )

        # Prepare where clause for time filtering
        where_clause = []
        if start_time is not None:
            where_clause.append(f"index >= Timestamp('{start_time}')")
        if end_time is not None:
            where_clause.append(f"index <= Timestamp('{end_time}')")
        where = " & ".join(where_clause) if where_clause else None

        frames: list[pd.DataFrame] = []

        if use_multiprocessing:
            logger.info(f"Loading {len(h5_keys)} groups using multiprocessing with {n_workers or mp.cpu_count() - 1} workers...")

            # Prepare arguments for worker processes
            worker_args = [(filepath, h5_key, where) for h5_key in h5_keys]

            # Determine number of workers
            if n_workers is None:
                n_workers = min(mp.cpu_count() - 1, len(h5_keys))
            else:
                n_workers = min(n_workers, len(h5_keys))

            try:
                # Use ProcessPoolExecutor for better control and Jupyter compatibility
                with ProcessPoolExecutor(max_workers=n_workers) as executor:
                    # Submit all tasks
                    future_to_key = {
                        executor.submit(_load_single_h5_group, args): args[1]
                        for args in worker_args
                    }

                    # Create a dictionary to store results by key for ordered processing
                    results_by_key = {}

                    # Collect results as they complete
                    for future in as_completed(future_to_key):
                        h5_key = future_to_key[future]
                        try:
                            df = future.result()
                            if not df.empty:
                                results_by_key[h5_key] = df
                        except Exception as e:
                            logger.error(f"Error loading {h5_key}: {str(e)}")
                            # Continue with other groups instead of failing completely

                # Add results to frames in the same order as h5_keys to maintain chronology
                for h5_key in h5_keys:
                    if h5_key in results_by_key:
                        logger.info(f"Appending {h5_key} to the frame list for concatanation.")
                        frames.append(results_by_key[h5_key])

            except Exception as e:
                logger.warning(f"Multiprocessing failed ({str(e)}), falling back to sequential loading...")
                use_multiprocessing = False

        # Sequential loading (fallback or when multiprocessing is disabled)
        if not use_multiprocessing:
            logger.info(f"Loading {len(h5_keys)} groups sequentially...")

            with pd.HDFStore(filepath, mode="r") as store:
                for h5_key in h5_keys:
                    try:
                        if where:
                            df = store.select(h5_key, where=where)
                        else:
                            df = store[h5_key]

                        if not df.empty:
                            frames.append(df)
                    except Exception as e:
                        logger.error(f"Error loading {h5_key}: {str(e)}")
                        continue

        if not frames:
            raise ValueError("No data was successfully loaded from any HDF5 group.")

        # ------------------------------------------------------------------
        #  Concatenate & restore original column order
        # ------------------------------------------------------------------
        logger.info(f"Concatenating {len(frames)} DataFrames...")
        df = pd.concat(frames, copy=False)

        # Ensure the DataFrame index is sorted
        if not df.index.is_monotonic_increasing:
            logger.info("Sorting DataFrame by datetime index after concatenation...")
            df.sort_index(inplace=True)

        logger.info(f"Successfully loaded {len(df):,} trades from {len(frames)} monthly groups.")

        side = df["side"] if "side" in df.columns else None
        side_values = side.values if side is not None else None

        return cls(df["timestamp"].values, df["price"].values, df["amount"].values,
                   side=side_values, dt_index=df.index)


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
    :param imb_max_run_signed: Optional longest signed imbalance run for each bar.
    :param vp_skew: Optional volume profile skew for each bar (positive = buy pressure above VWAP).
    :param vp_gini: Optional volume profile Gini coefficient for each bar (0 = concentrated, →1 = even).
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
    imb_max_run_signed: Optional[NDArray[np.int16]] = None    # 1D int16 array
    vp_skew: Optional[NDArray[np.float64]] = None             # 1D float64 array
    vp_gini: Optional[NDArray[np.float64]] = None             # 1D float64 array

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
                buy_imbalances_sum=self.buy_imbalances_sum[key] if self.buy_imbalances_sum is not None else None,
                imb_max_run_signed=self.imb_max_run_signed[key] if self.imb_max_run_signed is not None else None,
                vp_skew=self.vp_skew[key] if self.vp_skew is not None else None,
                vp_gini=self.vp_gini[key] if self.vp_gini is not None else None
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
        # Handle the case when the data tuple includes the new metrics
        if len(data) >= 14:  # Complete data tuple with all metrics
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
                buy_imbalances_sum=np.array(data[8], dtype=np.uint16),
                sell_imbalances_sum=np.array(data[9], dtype=np.uint16),
                cot_price_levels=np.array(data[10], dtype=np.int32),
                imb_max_run_signed=np.array(data[11], dtype=np.int16),
                vp_skew=np.array(data[12], dtype=np.float64),
                vp_gini=np.array(data[13], dtype=np.float64)
            )
        else:
            # Handle the case when the data tuple has the original fields only
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
                buy_imbalances_sum=np.array(data[8], dtype=np.uint16),
                sell_imbalances_sum=np.array(data[9], dtype=np.uint16),
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

    def memory_usage(self):
        """Calculate the approximate memory usage of this object in MB."""
        from pympler import asizeof
        import dataclasses
        total_memory = 0

        # Get field names from dataclass fields
        fields = [field.name for field in dataclasses.fields(self)]

        # Handle the main data attributes
        for attr in fields:
            if hasattr(self, attr):
                array = getattr(self, attr)
                if isinstance(array, np.ndarray) or isinstance(array, list) or isinstance(array, NumbaList):
                    try:
                        # Try using pympler
                        total_memory += sum(asizeof.asizeof(item) for item in array if item is not None)
                    except ValueError:
                        # Fallback for NumPy arrays with problematic memory layouts
                        if isinstance(array, np.ndarray):
                            total_memory += array.size * array.itemsize
                        else:
                            # Rough estimate for other types
                            total_memory += len(array) * 8  # Assume 8 bytes per object

        return total_memory / (1024 ** 2)

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


# --------
# utils
# --------
def _load_single_h5_group(args: Tuple[str, str, Optional[str]]) -> pd.DataFrame:
    """
    Helper function to load a single HDF5 group in a separate process.

    :param args: Tuple of (filepath, h5_key, where_clause)
    :returns: DataFrame with the loaded data
    """
    filepath, h5_key, where_clause = args

    try:
        with pd.HDFStore(filepath, mode="r") as store:
            if where_clause:
                df = store.select(h5_key, where=where_clause)
            else:
                df = store[h5_key]
        return df
    except Exception as e:
        # Return empty DataFrame with error info in case of failure
        logger.error(f"Failed to load {h5_key} from {filepath}: {str(e)}")
        return pd.DataFrame()


def _is_notebook_environment() -> bool:
    """
    Detect if we're running in a Jupyter notebook environment.

    :returns: True if in notebook, False otherwise
    """
    try:
        # Check for IPython
        from IPython import get_ipython
        if get_ipython() is not None:
            return True
    except ImportError:
        pass

    # Check for other notebook indicators
    return any('jupyter' in arg.lower() or 'ipython' in arg.lower() for arg in sys.argv)