import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Union, Tuple, Any
import datetime as dt
from finmlkit.utils.log import get_logger
import multiprocessing
from tqdm import tqdm
from .data_model import TradesData

logger = get_logger(__name__)


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


class AddTimeBarH5:
    """
    Builds and adds 1 sec TimeBar to trades h5
    """
    def __init__(self, h5_path: str, keys: list[str] = None):
        """

        :param h5_path: Path to trades h5 file
        :param keys: Optional list of keys for which to add TimeBars (eg. ["2022-01", "2022-05"]). If none, build all available month.
        """
        self.h5_path = h5_path
        self.keys = self._check_keys(keys)

    def _list_keys(self) -> list[str]:
        """
        List all available keys in the HDF5 file.

        :return: List of keys.
        """
        with pd.HDFStore(self.h5_path, mode='r') as store:
            return [k for k in store.keys() if k.startswith('/trades/')]

    def _check_keys(self, keys: list[str]):
        available_keys = self._list_keys()
        if keys:
            missing_keys = [key for key in keys if key not in available_keys]
            if missing_keys:
                raise KeyError(f"Missing keys: {missing_keys}\nAvailable keys: {available_keys}")
            return keys
        else:
            return available_keys

    def process_key(self, key: str, overwrite: bool = False) -> bool:
        """
        Process a single key to build and save 1-second time bars.

        :param key: The key to process (e.g., '/trades/2023-01')
        :param overwrite: Whether to overwrite existing time bar data for this key
        :return: True if successful, False otherwise
        """
        import pandas as pd
        from .kit import TimeBarKit
        from finmlkit.utils.log import get_logger
        logger = get_logger(__name__)

        # Derive timebar key from trade key
        timebar_key = key.replace('/trades/', '/klines/')
        month_key = key.split('/')[-1]

        # Check if time bars already exist
        with pd.HDFStore(self.h5_path, mode='r') as store:
            if timebar_key in store and not overwrite:
                logger.info(f"Time bars already exist for {month_key}. Skipping. Use overwrite=True to rebuild.")
                return False

        # Load trades data
        logger.info(f"Loading trades data for {month_key}...")
        trades_data = TradesData.load_trades_h5(self.h5_path, key=month_key)

        # Build 1-second time bars
        logger.info(f"Building 1-second time bars for {month_key}...")
        timebarkit = TimeBarKit(trades_data, period=pd.Timedelta(seconds=1))
        bars_df = timebarkit.build_ohlcv()

        # Save time bars to H5 store
        logger.info(f"Saving time bars for {month_key}...")
        with pd.HDFStore(self.h5_path, mode='a') as store:
            if timebar_key in store and overwrite:
                store.remove(timebar_key)

            store.put(
                key=timebar_key,
                value=bars_df,
                format='table',
                index=False
            )

            # Add metadata
            meta_key = timebar_key.replace('/klines/', '/klines_meta/')
            metadata = pd.Series({
                'record_count': len(bars_df),
                'first_timestamp': bars_df.index[0].value,
                'last_timestamp': bars_df.index[-1].value,
                'original_trades_key': key
            })

            if meta_key in store:
                store.remove(meta_key)
            store.put(meta_key, metadata, format='fixed')

        logger.info(f"Successfully added time bars for {month_key}. Created {len(bars_df)} bars.")
        return True

    def process_all(self, overwrite: bool = False) -> Dict[str, bool]:
        """
        Process all keys to build and save 1-second time bars.

        :param overwrite: Whether to overwrite existing time bar data
        :return: Dictionary mapping keys to success status
        """
        from finmlkit.utils.log import get_logger
        logger = get_logger(__name__)

        results = {}

        logger.info(f"Processing {len(self.keys)} months of trades data...")
        for key in tqdm(self.keys):
            logger.info(f"Processing {key}...")
            try:
                success = self.process_key(key, overwrite)
                results[key] = success
            except Exception as e:
                logger.error(f"Error processing {key}: {str(e)}")
                results[key] = False

        success_count = sum(1 for success in results.values() if success)
        logger.info(f"Processed {len(results)} keys with {success_count} successes.")

        return results


class TimeBarReader:
    """
    Reads time bars from an H5 file and allows resampling to larger timeframes.

    This class enables:
    - Reading 1-second time bars stored in an H5 file
    - Filtering by date range
    - Resampling to arbitrary timeframes (e.g., 5min, 1h, 1d)
    - Proper aggregation of OHLCV data
    - Correct calculation of VWAP for resampled periods
    """

    def __init__(self, h5_path: str):
        """
        Initialize the TimeBarReader with the path to the H5 file.

        :param h5_path: Path to the H5 file containing time bars
        """
        self.h5_path = h5_path

    def list_keys(self) -> List[str]:
        """
        List all available klines keys in the HDF5 file.

        :return: List of klines keys.
        """
        with pd.HDFStore(self.h5_path, mode='r') as store:
            return [k for k in store.keys() if k.startswith('/klines/')]

    def _list_time_range(self) -> Tuple[pd.Timestamp, pd.Timestamp]:
        """
        Determine the overall time range available in the H5 file.

        :return: Tuple of (first_timestamp, last_timestamp)
        """
        first_timestamp = None
        last_timestamp = None

        with pd.HDFStore(self.h5_path, mode='r') as store:
            meta_keys = [k for k in store.keys() if k.startswith('/klines_meta/')]

            for meta_key in meta_keys:
                metadata = store[meta_key]
                current_first = pd.to_datetime(metadata['first_timestamp'], unit='ns')
                current_last = pd.to_datetime(metadata['last_timestamp'], unit='ns')

                if first_timestamp is None or current_first < first_timestamp:
                    first_timestamp = current_first

                if last_timestamp is None or current_last > last_timestamp:
                    last_timestamp = current_last

        return first_timestamp, last_timestamp

    def _find_relevant_keys(self, start_time: Optional[pd.Timestamp] = None,
                           end_time: Optional[pd.Timestamp] = None) -> List[str]:
        """
        Find the keys that contain data within the specified time range.

        :param start_time: Start time for filtering, or None for earliest available
        :param end_time: End time for filtering, or None for latest available
        :return: List of keys that intersect with the specified time range
        """
        relevant_keys = []

        with pd.HDFStore(self.h5_path, mode='r') as store:
            for meta_key in [k for k in store.keys() if k.startswith('/klines_meta/')]:
                metadata = store[meta_key]
                key_first = pd.to_datetime(metadata['first_timestamp'], unit='ns')
                key_last = pd.to_datetime(metadata['last_timestamp'], unit='ns')

                # Check if key's time range intersects with requested time range
                if ((start_time is None or key_last >= start_time) and
                    (end_time is None or key_first <= end_time)):
                    klines_key = meta_key.replace('/klines_meta/', '/klines/')
                    relevant_keys.append(klines_key)

        return sorted(relevant_keys)

    def read(self,
             start_time: Optional[Union[str, pd.Timestamp, dt.datetime]] = None,
             end_time: Optional[Union[str, pd.Timestamp, dt.datetime]] = None,
             timeframe: Optional[str] = None) -> pd.DataFrame:
        """
        Read time bars from the H5 file, optionally filtering by time range and resampling.

        :param start_time: Start time for filtering (optional)
        :param end_time: End time for filtering (optional)
        :param timeframe: Timeframe for resampling (e.g., '5min', '1h', '1d', None for original 1s bars)
        :return: DataFrame with the requested time bars

        Examples:
            # Get all 1-second bars for a specific day
            reader = TimeBarReader('data.h5')
            df_1s = reader.read('2023-01-01', '2023-01-02')

            # Get 5-minute bars for a date range
            df_5min = reader.read('2023-01-01', '2023-01-31', timeframe='5min')

            # Get hourly bars for a specific month
            df_1h = reader.read('2023-01-01', '2023-01-31', timeframe='1h')

            # Get daily bars
            df_daily = reader.read(timeframe='1D')
        """
        # Normalize input parameters
        if isinstance(start_time, str):
            start_time = pd.Timestamp(start_time)
        if isinstance(end_time, str):
            end_time = pd.Timestamp(end_time)

        # Find relevant keys
        relevant_keys = self._find_relevant_keys(start_time, end_time)

        if not relevant_keys:
            logger.warning(f"No data found for time range: {start_time} to {end_time}")
            return pd.DataFrame()

        # Read data from H5 file
        dfs = []

        with pd.HDFStore(self.h5_path, mode='r') as store:
            for key in relevant_keys:
                # Load the full chunk - we'll filter in memory
                # This is more reliable than trying to use PyTables queries on index
                chunk = store[key]
                dfs.append(chunk)

        # Concatenate the chunks
        if not dfs:
            logger.warning("No data found after filtering")
            return pd.DataFrame()

        df = pd.concat(dfs)

        # Filter by time range if needed - using the index
        if start_time is not None or end_time is not None:
            df = df[(df.index >= start_time if start_time is not None else True) &
                   (df.index <= end_time if end_time is not None else True)]

        # Return original timeframe if no resampling requested
        if timeframe is None:
            return df

        # Resample to the requested timeframe
        return self._resample(df, timeframe)

    def _resample(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """
        Resample the data to the requested timeframe.

        :param df: DataFrame with 1s bars
        :param timeframe: Timeframe for resampling (e.g., '5min', '1h', '1d')
        :return: Resampled DataFrame
        """
        # Create resampler
        resampler = df.resample(timeframe)

        # Aggregate standard OHLCV values
        result = resampler.agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum',
            'bar_trades': 'sum',  # Sum the number of trades in the period
            'bar_median_trade_size': 'median',  # Take median of median trade sizes
        })

        # Calculate VWAP properly by accounting for volume
        vwap = (df['vwap'] * df['volume']).resample(timeframe).sum() / df['volume'].resample(timeframe).sum()
        result['vwap'] = vwap

        # Filter out rows with NaN values (potentially from missing periods)
        result = result.dropna(subset=['open', 'high', 'low', 'close'])

        return result
