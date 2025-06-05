import datetime as dt
from dataclasses import dataclass
from typing import Union, Optional, Tuple, Dict
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from numba.typed import List as NumbaList

from finmlkit.bar.utils import footprint_to_dataframe
from finmlkit.utils.log import get_logger
from .utils import comp_trade_side_vector, fast_sort_trades, merge_split_trades
import os
import multiprocessing

logger = get_logger(__name__)


class TradesData:
    """
    Class to preprocess trades data for bar building.

    This class handles standardization of column names, timestamp conversion,
    trade merging, and side inference for consistent processing across different
    data sources.
    """

    def __init__(self,
                 ts: NDArray, px: NDArray, qty: NDArray, *, id: NDArray = None,
                 is_buyer_maker: NDArray = None,
                 side = None,
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
        :param side: Optional Array Market order side information (-1: sell, 1: buy)
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

        self.data = pd.DataFrame({'timestamp': ts, 'price': px, 'amount': qty, 'id': id})
        self.is_buyer_maker = is_buyer_maker
        if side is not None:
            self.data['side'] = side
        self._orig_timestamp_unit = timestamp_unit if timestamp_unit else self.infer_timestamp_unit()
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
            if "side" not in self.data.columns:
                # If side info is not provided, infer it from trades data
                self._add_trade_side_info()

        # Add datetime_idx
        self.data.set_index(pd.to_datetime(self.data['timestamp'], unit='ns'), inplace=True)
        self.data.index.name = "datetime"
        logger.info("TradesData prepared successfully.")

    @property
    def orig_timestamp_unit(self) -> str:
        """
        Get the timestamp unit used for processing.

        :return: Timestamp unit string.
        """
        return self._orig_timestamp_unit

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

        # Check for gaps in trade IDs
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
            logger.info(f"Recorded {len(self.discontinuities)} trade ID discontinuities with corresponding time intervals.")
            self.missing_pct = cum_gap_size / len(self.data) * 100

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

        :param trades: DataFrame with trades to merge.
        :return: Merged DataFrame.
        """
        logger.info('Merging split trades (same timestamps) on same price level...')

        ts, px, am, side = merge_split_trades(
            self.data['timestamp'].values.astype(np.int64),
            self.data['price'].values.astype(np.float64),
            self.data['amount'].values.astype(np.float32),
            self.is_buyer_maker,
        )
        self.data = pd.DataFrame({
            'timestamp': ts,
            'price': px,
            'amount': am
        })
        if self.is_buyer_maker is not None:
            self.data['side'] = side

    def _convert_timestamps_to_ns(self):
        """
        Convert timestamps to nanosecond representation.

        :param trades: DataFrame with timestamps to convert.
        :param timestamp_unit: Unit of the timestamp values or None to infer.
        :return: The timestamp unit used for conversion.
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
        self.data['timestamp'].values[:] = np.multiply(self.data['timestamp'].values, factor, dtype=np.int64)

    def _apply_timestamp_resolution(self, proc_res: Optional[str]) -> None:
        """
        Apply processing resolution to timestamps if specified.

        :param trades: DataFrame with timestamps already converted to nanoseconds.
        :param timestamp_unit: Current timestamp unit (should be 'ns' at this point).
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

        :param trades: DataFrame to process.
        :returns: None - modifies the trades DataFrame in place to include a 'side' column.
        """
        logger.info("No trade side information found. Inferring trade side from price movements.")
        self.data['side'] = comp_trade_side_vector(self.data['price'].values)

    def infer_timestamp_unit(self) -> str:
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
        """
        Persist the raw trades to an on-disk HDF5 store.
        The data of **each calendar month** lives under ``/trades/YYYY-MM`` in the file.

        - When adding new monthly data, it will be stored in a new group.
        - When adding data for an existing month, you can either append to it or overwrite it with confirmation.

        :param filepath: Destination `.h5` file. The parent directories are created automatically when missing.
        :param month_key: Override the key of the form ``"YYYY-MM"``. When ``None`` the key is derived from the first timestamp of ``self.data``.
        :param complib: Compression backend used by PyTables. Default is ``blosc:zstd``.
        :param complevel: Compression level. Default is 5.
        :param mode: File mode – ``"a"`` to create or append, ``"w"`` to start fresh. Default is ``"a"``.
        :param chunksize: Row chunk size used by PyTables when writing large frames. Default is 1000000.
        :param overwrite_month: If True and the month data exists, prompts for confirmation to overwrite. Default is True.

        :returns: The full key used inside the store, e.g. ``"/trades/2025-02"``.
        :raises: ValueError if user declines to overwrite existing data.
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
    def load_h5(
        cls,
        filepath: str,
        *,
        key: Optional[str] = None,
        start_time: Optional[Union[str, pd.Timestamp]] = None,
        end_time: Optional[Union[str, pd.Timestamp]] = None,
    ) -> "TradesData":
        """Load trades from *filepath*.

        Three usage modes exist:
        1. ``key`` only – load the full monthly partition ``/trades/<key>``.
        2. ``start_time`` / ``end_time`` – assemble the minimal set of monthly
           groups touching the range, slice **at read time** for maximum speed.
        3. Combination – constrain selection *within* the chosen "key".

        :returns: `TradesData`
        """
        # ------------------------------------------------------------------
        #  Normalise temporal boundaries
        # ------------------------------------------------------------------
        if isinstance(start_time, str):
            start_time = pd.Timestamp(start_time)
        if isinstance(end_time, str):
            end_time = pd.Timestamp(end_time)

        frames: list[pd.DataFrame] = []

        logger.info(f"Loading trades from {filepath}...")
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

            # Retrieve each group with on‑store slicing ----------------------------------------
            where_clause = []
            if start_time is not None:
                where_clause.append(f"index >= Timestamp('{start_time}')")
            if end_time is not None:
                where_clause.append(f"index <= Timestamp('{end_time}')")
            where = " & ".join(where_clause) if where_clause else None

            for h5_key in h5_keys:
                if where:
                    frames.append(store.select(h5_key, where=where))
                else:
                    frames.append(store[h5_key])

        # ------------------------------------------------------------------
        #  Concatenate & restore original column order
        # ------------------------------------------------------------------
        df = pd.concat(frames, copy=False)

        # Ensure the DataFrame index is sorted
        if not df.index.is_monotonic_increasing:
            logger.warning("Dataframe index is not indexing after concat. Sorting the DataFrame index...")
            df.sort_index(inplace=True)


        side = df["side"] if "side" in df.columns else None
        return cls(df["timestamp"].values, df["price"].values, df["amount"].values, side=side.values)


def _find_gaps(key: str, filepath: str, max_gap: pd.Timedelta) -> Tuple[str, list[Tuple[pd.Timestamp, pd.Timedelta]]]:
    """
    Find gaps in the trades data for a specific key.

    :param key: HDF5 key to inspect.
    :param filepath: Path to the HDF5 file.
    :param max_gap: Maximum allowable gap between consecutive timestamps.
    :return: Tuple containing the key and a list of tuples, each with (gap timestamp, gap size).
    """
    with pd.HDFStore(filepath, mode='r') as store:
        df = store[key]
        diff_series = df.index.to_series().diff()
        gap_mask = diff_series > max_gap
        gap_timestamps = df.index[gap_mask].tolist()
        gap_sizes = diff_series[gap_mask].tolist()

        # Combine timestamps and gap sizes into tuples
        gaps_with_sizes = list(zip(gap_timestamps, gap_sizes))

        return key, gaps_with_sizes


class H5Inspector:
    """
    Class to inspect HDF5 files containing trades data.

    This class provides methods to list available keys, check metadata,
    and retrieve basic statistics about the trades data stored in HDF5 format.
    """

    def __init__(self, filepath: str):
        """
        Initialize the H5Inspector with the path to the HDF5 file.

        :param filepath: Path to the HDF5 file.
        """
        self.filepath = filepath

    def list_keys(self) -> list[str]:
        """
        List all available keys in the HDF5 file.

        :return: List of keys.
        """
        with pd.HDFStore(self.filepath, mode='r') as store:
            return [k for k in store.keys() if k.startswith('/trades/')]

    def get_metadata(self, key: str) -> Dict[str, any]:
        """
        Get metadata for a specific key in the HDF5 file.

        :param key: Key to retrieve metadata for (Eg.: /trades/2023-02)
        :return: Metadata dictionary.
        """
        with pd.HDFStore(self.filepath, mode='r') as store:
            if key not in store.keys():
                raise KeyError(f"Key '{key}' not found in the store.")
            meta_key = key.replace('/trades/', '/meta/')
            return store[meta_key].to_dict()

    def get_integrity_info(self, key: str) -> Optional[pd.DataFrame]:
        """
        Get data integrity information for a specific key in the HDF5 file.
        This retrieves discontinuity information stored during the save_h5 process.

        :param key: Key to retrieve integrity information for (e.g., '/trades/2023-01').
        :return: DataFrame with discontinuity information or None if no integrity issues were found.
        """
        with pd.HDFStore(self.filepath, mode='r') as store:
            if key not in store.keys():
                raise KeyError(f"Key '{key}' not found in the store.")

            integrity_key = key.replace('/trades/', '/integrity/')

            if integrity_key in store:
                disc_df = store[integrity_key]

                return disc_df
            else:
                return None

    def get_statistics(self, key: str) -> Dict[str, any]:
        """
        Get basic statistics for a specific key in the HDF5 file.

        :param key: Key to retrieve statistics for.
        :return: Statistics dictionary.
        """
        with pd.HDFStore(self.filepath, mode='r') as store:
            if key not in store.keys():
                raise KeyError(f"Key '{key}' not found in the store.")
            df = store[key]
            return {
                'record_count': len(df),
                'first_timestamp': df['timestamp'].min(),
                'last_timestamp': df['timestamp'].max(),
                'price_range': (df['price'].min(), df['price'].max()),
                'amount_range': (df['amount'].min(), df['amount'].max())
            }

    def inspect_gaps(self, max_gap: pd.Timedelta = pd.Timedelta(minutes=1), processes: int = 4) -> Dict[str, list[tuple[pd.Timestamp, pd.Timedelta]]]:
        """
        Inspect gaps in trades data across all keys in the HDF5 file.

        :param max_gap: Maximum allowable gap between consecutive timestamps.
        :param processes: Number of processes to use for multiprocessing.
        :return: Dictionary with keys as HDF5 groups and values as lists of gap timestamps.
        """
        keys = self.list_keys()

        with multiprocessing.Pool(processes=processes) as pool:
            results = pool.starmap(_find_gaps, [(key, self.filepath, max_gap) for key in keys])

        return dict(results)

    def get_integrity_summary(self, verbose=True) -> Dict[str, Dict]:
        """
        Generate a summary of data integrity issues across all tables in the HDF5 file.

        This function identifies tables with integrity issues (data_integrity_ok=False),
        collects statistics about the issues (missing percentage, etc.), and retrieves
        the detailed discontinuity information for affected tables.

        :param verbose: Whether to print the results to console
        :return: Dictionary with keys as HDF5 groups and values as dictionaries containing:
                 - 'metadata': Basic metadata about the table including integrity flags
                 - 'discontinuities': DataFrame with detailed discontinuity information (if available)
                 - Or None if no integrity issues are found
        """
        result = {}
        all_ok = True

        # Get all available trade keys
        keys = self.list_keys()

        with pd.HDFStore(self.filepath, mode='r') as store:
            # Check if there are any keys first
            if not keys:
                return None

            # First pass: collect metadata for all tables
            for key in keys:
                meta_key = key.replace('/trades/', '/meta/')
                integrity_key = key.replace('/trades/', '/integrity/')

                if meta_key in store:
                    metadata = store[meta_key].to_dict()
                    # Check if the integrity flag is False
                    if 'data_integrity_ok' in metadata and metadata['data_integrity_ok'] is False:
                        all_ok = False
                        month_key = key.split('/')[-1]  # Extract the month part (e.g., '2023-01')
                        result[month_key] = {
                            'metadata': metadata,
                            'key': key
                        }

                        # Add discontinuity information if available
                        if integrity_key in store:
                            disc_df = store[integrity_key]
                            result[month_key]['discontinuities'] = disc_df
                        else:
                            result[month_key]['discontinuities'] = None

        if all_ok:
            logger.info("All data passed integrity checks. No issues found.")
            return None

        # Count summary statistics
        issue_count = len(result)
        avg_missing_pct = sum(
            info['metadata'].get('missing_pct', 0) for info in result.values()) / issue_count if issue_count > 0 else 0

        logger.info(f"Found {issue_count} tables with data integrity issues.")
        logger.info(f"Average missing data percentage: {avg_missing_pct:.2f}%")

        if verbose:
            # Process the results
            if result:
                print(f"Found {len(result)} months with data integrity issues:")
                for month, info in result.items():
                    print("\n===================================================================")
                    print(f"Month: {month}")
                    print("=====================================================================")
                    print(f"Missing data: {info['metadata']['missing_pct']:.2f}%")
                    print(f"First timestamp: {pd.to_datetime(info['metadata']['first_timestamp'], unit='ns')}")
                    print(f"Last timestamp: {pd.to_datetime(info['metadata']['last_timestamp'], unit='ns')}")

                    if info['discontinuities'] is not None:
                        print(f"Number of discontinuities: {len(info['discontinuities'])}")
                        print("\nDiscontinuities:")
                        print(info['discontinuities'].T)
            else:
                print("No data integrity issues found!")

        return result


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

