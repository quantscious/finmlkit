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
    r"""Utility class for inspecting and analyzing HDF5 files containing trades data with comprehensive metadata access.

    This class provides a complete toolkit for examining HDF5 stores created by :class:`TradesData`, enabling users to
    explore available data, assess data quality, retrieve statistics, and identify potential issues across monthly
    partitioned trade datasets. It serves as an essential diagnostic tool for large-scale financial data management.

    The inspector is designed to work with the HDF5 structure created by :meth:`TradesData.save_h5`, where trades data
    is organized into monthly partitions under ``/trades/YYYY-MM`` groups, with corresponding metadata stored under
    ``/meta/YYYY-MM`` and integrity information under ``/integrity/YYYY-MM``.

    Key capabilities include:

    - **Data Discovery**: List all available monthly partitions and their temporal coverage
    - **Metadata Access**: Retrieve comprehensive metadata including record counts, timestamp ranges, and integrity flags
    - **Integrity Analysis**: Access detailed information about trade ID discontinuities, missing data percentages,
      and temporal gaps that may indicate data quality issues
    - **Statistical Overview**: Compute basic statistics for price and volume distributions across time periods
    - **Gap Detection**: Identify temporal discontinuities exceeding specified thresholds using multiprocessing
    - **Integrity Reporting**: Generate comprehensive summaries of data quality issues across entire datasets

    The class leverages the metadata structure to provide fast operations without loading full datasets into memory,
    making it suitable for inspecting multi-terabyte trade databases. For gap analysis and integrity checks on
    large datasets, multiprocessing is employed to parallelize operations across monthly partitions.

    **Data Integrity Metrics**: The inspector can identify several types of data quality issues:

    - **Trade ID Gaps**: Missing sequential trade IDs indicating potential data loss
    - **Temporal Discontinuities**: Time gaps exceeding normal market hours or trading halts
    - **Missing Data Percentage**: Quantitative measure of data completeness based on trade ID sequences

    .. note::
        This class assumes HDF5 files follow the structure created by :class:`TradesData`. For files created
        with different schemas, some methods may not function correctly or may raise KeyError exceptions.

    .. note::
        Gap detection with multiprocessing can be memory-intensive for very large datasets. Consider adjusting
        the ``processes`` parameter based on available system resources and dataset sizes.

    Args:
        filepath (str): Path to the HDF5 file containing trades data. File must be readable and follow
            the expected monthly partition structure.

    Raises:
        FileNotFoundError: If the specified HDF5 file does not exist.
        PermissionError: If the file cannot be accessed due to permission restrictions.

    See Also:

        - :class:`finmlkit.bar.data_model.TradesData`: Creates the HDF5 files that this class inspects.
        - :meth:`finmlkit.bar.data_model.TradesData.save_h5`: Method that creates the HDF5 structure.
        - :meth:`finmlkit.bar.data_model.TradesData.load_trades_h5`: Complementary loading functionality.

    References:
        .. _`HDF5 for Python`: https://docs.h5py.org/en/stable/
        .. _`Pandas HDFStore`: https://pandas.pydata.org/docs/reference/api/pandas.HDFStore.html
    """

    def __init__(self, filepath: str):
        """
        Initialize the H5Inspector with the path to the HDF5 file.

        :param filepath: Path to the HDF5 file.
        """
        self.filepath = filepath

    def list_keys(self) -> list[str]:
        r"""List all available trade keys in the HDF5 file.

        Scans the HDF5 store for all groups under the ``/trades/`` hierarchy,
        returning a sorted list of available monthly partitions.

        :return: List of trade keys in format ``['/trades/YYYY-MM', ...]``.
        :raises FileNotFoundError: If the HDF5 file does not exist.
        :raises KeyError: If the file exists but has no readable trade groups.
        """
        with pd.HDFStore(self.filepath, mode='r') as store:
            return [k for k in store.keys() if k.startswith('/trades/')]

    def get_metadata(self, key: str) -> Dict[str, any]:
        r"""Retrieve comprehensive metadata for a specific monthly partition.

        Returns metadata stored during the save process, including record counts,
        timestamp ranges, data integrity flags, and missing data percentages.

        :param key: HDF5 key for the target month (e.g., '/trades/2023-02').
        :return: Dictionary containing metadata fields:

            - 'record_count': Number of trades in the partition
            - 'first_timestamp': Earliest timestamp (nanoseconds since epoch)
            - 'last_timestamp': Latest timestamp (nanoseconds since epoch)
            - 'data_integrity_ok': Boolean flag indicating data quality
            - 'missing_pct': Percentage of missing trades based on ID gaps
        :raises KeyError: If the specified key does not exist in the store.
        """
        with pd.HDFStore(self.filepath, mode='r') as store:
            if key not in store.keys():
                raise KeyError(f"Key '{key}' not found in the store.")
            meta_key = key.replace('/trades/', '/meta/')
            return store[meta_key].to_dict()

    def get_integrity_info(self, key: str) -> Optional[pd.DataFrame]:
        r"""Retrieve detailed data integrity information for a specific monthly partition.

        Returns discontinuity details stored during preprocessing, including trade ID gaps,
        timestamps of missing data periods, and time intervals for each discontinuity.

        :param key: HDF5 key for the target month (e.g., '/trades/2023-01').
        :return: DataFrame with columns:

            - 'start_id': Trade ID before the gap
            - 'end_id': Trade ID after the gap
            - 'missing_ids': Number of missing trade IDs
            - 'pre_gap_time_str': Timestamp before gap (string format)
            - 'post_gap_time_str': Timestamp after gap (string format)
            - 'time_interval_str': Duration of the gap (string format)

            Returns None if no integrity issues were detected.
        :raises KeyError: If the specified key does not exist in the store.
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
        r"""Compute basic statistical measures for a specific monthly partition.

        Loads the trade data and calculates summary statistics including record counts,
        timestamp ranges, and price/volume distributions.

        :param key: HDF5 key for the target month.
        :return: Dictionary containing statistical measures:

            - 'record_count': Total number of trade records
            - 'first_timestamp': Earliest timestamp in the dataset
            - 'last_timestamp': Latest timestamp in the dataset
            - 'price_range': Tuple of (minimum_price, maximum_price)
            - 'amount_range': Tuple of (minimum_amount, maximum_amount)
        :raises KeyError: If the specified key does not exist in the store.

        .. note::
            This method loads the full dataset into memory and may be slow for large partitions.
            Consider using :meth:`get_metadata` for basic counts and ranges when available.
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
        r"""Identify temporal gaps exceeding specified thresholds across all monthly partitions.

        Uses multiprocessing to parallelize gap detection across partitions. Gaps are identified
        by computing time differences between consecutive trades and flagging those exceeding
        the ``max_gap`` threshold.

        :param max_gap: Maximum allowable gap between consecutive timestamps. Default: 1 minute.
        :param processes: Number of worker processes for parallel processing. Default: 4.
        :return: Dictionary mapping HDF5 group names to lists of gap information tuples:
            Each tuple contains (gap_timestamp, gap_duration) for gaps exceeding the threshold.
        :raises ValueError: If max_gap is not a valid Timedelta or processes < 1.

        .. note::
            Gap detection loads full datasets into memory. For very large files, consider
            processing monthly partitions individually or increasing available system memory.

        """
        keys = self.list_keys()

        with multiprocessing.Pool(processes=processes) as pool:
            results = pool.starmap(_find_gaps, [(key, self.filepath, max_gap) for key in keys])

        return dict(results)

    def get_integrity_summary(self, verbose=True) -> Dict[str, Dict]:
        r"""Generate comprehensive summary of data integrity issues across the entire HDF5 store.

        Analyzes all monthly partitions to identify data quality problems, providing both
        aggregate statistics and detailed discontinuity information where available.

        :param verbose: If True, prints detailed summary to console. Default: True.
        :return: Dictionary with month keys mapping to integrity information dictionaries:
            Each value contains:
            - 'metadata': Complete metadata including integrity flags and missing percentages
            - 'discontinuities': DataFrame with detailed gap information (if available)
            - 'key': Original HDF5 key for the partition
            Returns None if all data passes integrity checks.

        .. note::
            This method scans metadata for all partitions but only loads detailed discontinuity
            information for months with identified issues, making it efficient for large stores.
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
    r"""Utility class for building and persisting 1-second time bars from trades data stored in HDF5 format.

    This class provides a streamlined workflow for converting raw trades data into structured time bars,
    extending HDF5 stores created by :class:`TradesData` with standardized OHLCV (Open, High, Low, Close, Volume)
    bars at 1-second intervals. It serves as a preprocessing component for financial analysis pipelines that
    require consistent temporal aggregation of high-frequency trading data.

    The class operates on HDF5 files with monthly trade partitions (``/trades/YYYY-MM``) and creates
    corresponding time bar partitions (``/klines/YYYY-MM``) with associated metadata (``/klines_meta/YYYY-MM``).
    This approach maintains the same organizational structure while adding derived datasets optimized for
    time-series analysis and modeling.

    **Workflow and Data Organization:**

    The class follows this processing pipeline:

    1. **Discovery**: Identify available monthly trade partitions in the source HDF5 file
    2. **Loading**: Use :meth:`TradesData.load_trades_h5` to retrieve preprocessed trades for each month
    3. **Aggregation**: Apply :class:`TimeBarKit` to construct 1-second time bars with full OHLCV features
    4. **Storage**: Persist bars to ``/klines/`` hierarchy with metadata for fast access
    5. **Validation**: Track processing success/failure for each monthly partition

    The resulting time bars provide a consistent temporal grid suitable for:

    - Technical analysis and indicator computation
    - Machine learning feature engineering
    - Risk management and portfolio analytics
    - But mainly for accessing **coarse sampling** from high-frequency data (e.g., accesing daily statistics quickly)

    **Considerations:**

    - Processing is performed sequentially by month to manage memory usage for large datasets
    - Each month's bars are stored as separate HDF5 tables for efficient partial loading
    - Metadata storage enables fast discovery without loading full datasets
    - Overwrite protection prevents accidental data loss during reprocessing

    .. important::
        This class assumes the source HDF5 file follows the structure created by :class:`finmlkit.bar.data_model.TradesData`.
        The time bars are built using 1-second intervals, which provides a good balance between temporal
        resolution and data reduction for most financial analysis applications.

    .. tip::
        For very active trading pairs, 1-second bars may still contain significant noise. Consider
        further aggregation (e.g., 1-minute bars) for certain analysis types or implement alternative
        bar types (tick, volume, or imbalance bars) using :class:`BarBuilderBase` subclasses.

    .. note::
        This enables the quick construction and retention of simple aggregated OHLCV bars with specified frequency.
        If you want more intra-bar features (e.g., directional, size, or footprint features) you should use
        :class:`finmlkit.bar.kit.TimeBarKit` directly to build bars from trades data.

    Args:
        h5_path (str): Path to the HDF5 file containing trades data. Must be readable and writable.
        keys (list[str], optional): Specific monthly keys to process (e.g., ["2022-01", "2022-05"]).
            If None, processes all available monthly partitions in the file.

    Raises:
        KeyError: If specified keys are not found in the source HDF5 file.
        FileNotFoundError: If the HDF5 file does not exist.
        PermissionError: If the file cannot be accessed for reading or writing.

    Examples:
        Process all months in an HDF5 file:

        >>> # doctest: +SKIP
        >>> processor = AddTimeBarH5('trades_2023.h5')
        >>> results = processor.process_all(overwrite=False)
        >>> success_count = sum(results.values())
        >>> print(f"Successfully processed {success_count}/{len(results)} months")

        Process specific months with overwrite:

        >>> # doctest: +SKIP
        >>> processor = AddTimeBarH5('trades_2023.h5', keys=['2023-03', '2023-04'])
        >>> for key in processor.keys:
        ...     success = processor.process_key(key, overwrite=True)
        ...     print(f"{key}: {'Success' if success else 'Failed'}")

        Batch processing workflow:

        >>> # doctest: +SKIP
        >>> import pandas as pd
        >>> processor = AddTimeBarH5('large_dataset.h5')
        >>> results = processor.process_all()
        >>>
        >>> # Check results and identify any failures
        >>> failed_keys = [k for k, success in results.items() if not success]
        >>> if failed_keys:
        ...     print(f"Failed to process: {failed_keys}")

    See Also:

        :class: `TimeBarReader`: For reading and analyzing the time bars (generated by this class) from HDF5 files.
        :class:`finmlkit.bar.data_model.TradesData`: Creates the source HDF5 files with trades data.
        :meth:`finmlkit.bar.data_model.TradesData.save_h5`: Saves trades data to HDF5 format which is used by this class to add time bars to it.
        :class:`finmlkit.bar.kit.TimeBarKit`: The underlying time bar construction engine.
        :class:`finmlkit.bar.base.BarBuilderBase`: Base class for bar construction strategies.

    References:
        .. _`Time Bar Construction in Financial ML`: https://www.wiley.com/en-us/Advances+in+Financial+Machine+Learning-p-9781119482086
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
        r"""Process a single monthly partition to build and save 1-second time bars.

        Loads trades data for the specified month, constructs time bars using :class:`TimeBarKit`,
        and persists the results to the HDF5 file under the ``/klines/`` hierarchy.

        :param key: The trades key to process (format: '/trades/YYYY-MM' or 'YYYY-MM').
        :param overwrite: Whether to overwrite existing time bar data for this partition. Default: False.
        :returns: True if processing completed successfully, False if skipped or failed.

        .. note::
            Processing time scales with the number of trades in the month. For very active trading pairs,
            expect several minutes per month on typical hardware. Memory usage peaks during bar construction
            but is released after each month completes.

        Examples:
            >>> # doctest: +SKIP
            >>> processor = AddTimeBarH5('trades.h5')
            >>> success = processor.process_key('/trades/2023-06', overwrite=True)
            >>> if success:
            ...     print("Time bars created successfully")
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
        r"""Process all configured monthly partitions to build and save 1-second time bars.

        Iterates through all keys (either specified during initialization or auto-discovered)
        and processes each month sequentially. Provides comprehensive logging and error handling
        to ensure robust batch processing of large datasets.

        :param overwrite: Whether to overwrite existing time bar data for all partitions. Default: False.
        :returns: Dictionary mapping partition keys to processing success status (True/False).
            Keys are in format '/trades/YYYY-MM' and values indicate whether processing completed successfully.

        .. note::
            Processing is performed sequentially to manage memory usage. For very large datasets,
            monitor system resources and consider processing subsets if memory constraints arise.
            Failed partitions can be reprocessed individually using :meth:`process_key`.

        Examples:
            >>> # doctest: +SKIP
            >>> processor = AddTimeBarH5('annual_trades.h5')
            >>> results = processor.process_all(overwrite=False)
            >>>
            >>> # Analyze results
            >>> total_processed = len(results)
            >>> successful = sum(results.values())
            >>> print(f"Processed {successful}/{total_processed} months successfully")
            >>>
            >>> # Identify and retry failed months
            >>> failed_months = [k for k, success in results.items() if not success]
            >>> for month in failed_months:
            ...     processor.process_key(month, overwrite=True)  # Retry with overwrite
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
    r"""Reader class for time bar data stored in HDF5 format with advanced resampling capabilities.

    This class provides a comprehensive interface for accessing and transforming time bar data created by
    :class:`AddTimeBarH5`, enabling efficient querying, filtering, and resampling of high-frequency financial
    time series. It serves as the primary access layer for time bar analysis workflows, supporting both
    raw 1-second bars and dynamically resampled timeframes for various analytical purposes.

    The reader is designed to work seamlessly with the HDF5 structure created by the time bar processing
    pipeline, where 1-second bars are stored under ``/klines/YYYY-MM`` groups with metadata under
    ``/klines_meta/YYYY-MM``. This organization enables efficient time-range queries across large datasets
    without loading unnecessary data into memory.

    **Core Functionalities:**

    - **Time Range Filtering**: Efficiently identify and load only the monthly partitions intersecting
      with requested time ranges, minimizing memory usage and I/O operations.

    - **Flexible Resampling**: Transform 1-second bars into arbitrary timeframes (e.g., 5min, 1h, 1d)
      with mathematically correct aggregation of OHLCV data and volume-weighted recalculation of derived metrics.

    - **Metadata-Driven Discovery**: Leverage stored metadata for fast range queries without scanning
      full datasets, enabling sub-second response times for time range validation.

    **Performance Optimizations:**

    The reader employs several strategies for efficient large-scale data access:

    - **Lazy Loading**: Only relevant monthly partitions are identified and loaded based on time range intersection
    - **Vectorized Operations**: Resampling uses pandas' optimized groupby operations with pre-computed time groupers
    - **Memory Management**: Data is processed in monthly chunks and concatenated only when necessary
    - **Index Optimization**: Time filtering leverages datetime indexes for fast range selection

    .. important::
        :class:`TimeBarReader` assumes data integrity and proper temporal ordering within each monthly partition.
        The input H5 file must be generated with :class:`finmlkit.bar.data_model.TradesData` and :class:`AddTimeBarH5`.

    .. note::
        This enables the quick construction and retention of simple aggregated OHLCV bars with specified frequency.
        If you want more intra-bar features (e.g., directional, size, or footprint features) you should use
        :class:`finmlkit.bar.kit.TimeBarKit` directly to build bars from trades data.

    .. tip::
        Resampling to very large timeframes (e.g., monthly) from 1-second data can be memory-intensive.
        For such cases, consider intermediate aggregation steps or processing smaller time ranges iteratively.

    Args:
        h5_path (str): Path to the HDF5 file containing time bar data. Must be readable and contain
            data structure created by :class:`AddTimeBarH5`.

    Raises:
        FileNotFoundError: If the specified HDF5 file does not exist.
        PermissionError: If the file cannot be accessed due to permission restrictions.
        KeyError: If the file exists but lacks the expected klines structure.

    Examples:
        Basic time bar reading with range filtering:

       >>> # doctest: +SKIP
       >>> from finmlkit.bar.io import TimeBarReader
       >>> reader = TimeBarReader('trades_2023.h5')
       >>>
       >>> # Get all 1-second bars for a specific day
       >>> bars_1s = reader.read('2023-01-15', '2023-01-15')
       >>> len(bars_1s)  # doctest: +SKIP
       86400
       >>> # doctest: +SKIP
       >>> # Get 5-minute bars for a week
       >>> bars_5min = reader.read('2023-01-15', '2023-01-21', timeframe='5min')
       >>> bars_5min.columns.tolist()  # doctest: +SKIP
       ['open', 'high', 'low', 'close', 'volume', 'trades', 'vwap', 'median_trade_size']

    Advanced resampling workflows:

       >>> # doctest: +SKIP
       >>> # Get hourly bars with proper VWAP calculation
       >>> hourly = reader.read('2023-01-01', '2023-01-31', timeframe='1h')
       >>> # Verify VWAP is volume-weighted across the resampled period
       >>> print(f"First hourly bar VWAP: {hourly['vwap'].iloc[0]:.2f}")  # doctest: +SKIP
       First hourly bar VWAP: 16750.25

    Data discovery and range validation:

       >>> # doctest: +SKIP
       >>> available_months = reader.list_keys()
       >>> print(f"Available months: {len(available_months)}")  # doctest: +SKIP
       Available months: 12
       >>>
       >>> # Check overall time coverage
       >>> start, end = reader._list_time_range()  # doctest: +SKIP
       >>> print(f"Data spans from {start.date()} to {end.date()}")  # doctest: +SKIP
       Data spans from 2023-01-01 to 2023-12-31

    See Also:

        - :class:`AddTimeBarH5`: Creates the HDF5 time bar files that this reader accesses.
        - :class:`finmlkit.bar.data_model.TradesData`: Underlying trades data structure for the bar construction process.
        - :class:`finmlkit.bar.kit.TimeBarKit`: Time bar construction engine used by the processing pipeline.
        - :class:`H5Inspector`: Complementary utility for HDF5 file inspection and data quality assessment.
    """

    def __init__(self, h5_path: str):
        """
        Initialize the TimeBarReader with the path to the H5 file.

        :param h5_path: Path to the H5 file containing time bars
        """
        self.h5_path = h5_path

    def list_keys(self) -> List[str]:
        r"""List all available time bar keys in the HDF5 file.

        Scans the HDF5 store for all klines groups, providing visibility into available
        monthly partitions for time range planning and data discovery.

        :return: List of klines keys in format ``['/klines/YYYY-MM', ...]``, sorted chronologically.
        :raises FileNotFoundError: If the HDF5 file does not exist.
        :raises PermissionError: If the file cannot be accessed for reading.

        Examples:

            >>> # doctest: +SKIP
            >>> reader = TimeBarReader('data.h5')
            >>> keys = reader.list_keys()
            >>> print(f"Found {len(keys)} monthly partitions")
            Found 12 monthly partitions
        """
        with pd.HDFStore(self.h5_path, mode='r') as store:
            return [k for k in store.keys() if k.startswith('/klines/')]

    def _list_time_range(self) -> Tuple[pd.Timestamp, pd.Timestamp]:
        r"""Determine the overall temporal coverage available in the HDF5 file by scanning metadata.

        Efficiently discovers the time span of available data without loading full datasets,
        enabling quick validation of data availability for query planning.

        :returns: Tuple of (earliest_timestamp, latest_timestamp) across all monthly partitions.
        :raises ValueError: If no klines metadata is found in the file.

        .. note::
            This method relies on metadata stored during the bar creation process and provides
            sub-second response times even for very large datasets.
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
        r"""Identify HDF5 keys containing data that intersects with the specified time range.

        Uses metadata-driven discovery to minimize I/O by identifying only the monthly partitions
        that contain relevant data, avoiding unnecessary loading of non-intersecting partitions.

        :param start_time: Start boundary for range intersection test. None indicates no lower bound.
        :param end_time: End boundary for range intersection test. None indicates no upper bound.
        :returns: Sorted list of klines keys that intersect with the specified time range.

        .. note::
            The intersection logic is inclusive on both ends, ensuring that monthly partitions
            containing any data within the range are included in the result set.
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
        r"""Read time bars from HDF5 storage with optional time filtering and resampling.

        This method provides the primary interface for accessing time bar data, supporting flexible
        time range specification and dynamic resampling to arbitrary timeframes. The implementation
        optimizes for both small targeted queries and large-scale data processing workflows.

        **Time Range Handling:**

        - **Inclusive Ranges**: Both start_time and end_time are treated as inclusive boundaries
        - **Date Normalization**: Date strings without time components are expanded to full day ranges
        - **Boundary Correction**: End dates are automatically extended to include the entire final day

        **Resampling Process:**

        When a timeframe is specified, the method applies mathematically correct aggregation:

        1. Groups 1-second bars by the requested timeframe using vectorized floor operations
        2. Applies OHLCV aggregation rules (first, max, min, last, sum)
        3. Recalculates volume-weighted metrics (VWAP, median trade size) preserving statistical properties
        4. Filters out empty periods to maintain data density

        :param start_time: Start time for filtering (inclusive). Accepts string, Timestamp, or datetime.
            If None, starts from earliest available data.
        :param end_time: End time for filtering (inclusive). If provided as date-only string,
            automatically extends to end of day. If None, includes all data through latest available.
        :param timeframe: Target resampling timeframe using pandas offset aliases
            (e.g., '5min', '1h', '1D', '1W'). If None, returns original 1-second bars.
        :returns: DataFrame with datetime index and columns: open, high, low, close, volume, trades, vwap, median_trade_size.
            Empty DataFrame if no data found in specified range.

        :raises ValueError: If start_time > end_time or timeframe format is invalid.
        :raises KeyError: If required data partitions are missing from the HDF5 file.

        .. note::
            For daily or longer timeframes, incomplete final periods (e.g., partial trading days)
            are automatically excluded to prevent misleading aggregations in analysis workflows.

        Examples:

            Reading specific time ranges:

            >>> reader = TimeBarReader('crypto_data.h5')  # doctest: +SKIP
            >>>
            >>> # Get all 1-second bars for Bitcoin on January 15, 2023
            >>> btc_1s = reader.read('2023-01-15', '2023-01-15')  # doctest: +SKIP
            >>> print(f"Retrieved {len(btc_1s):,} 1-second bars")  # doctest: +SKIP
            Retrieved 86,400 1-second bars
            >>>
            >>> # Get 5-minute bars for the first week of January
            >>> btc_5min = reader.read('2023-01-01', '2023-01-07', timeframe='5min')  # doctest: +SKIP
            >>> print(f"5-min bars: {len(btc_5min)}")  # doctest: +SKIP
            5-min bars: 2016

            Resampling to various timeframes:

            >>> # Hourly bars with volume-weighted VWAP
            >>> hourly = reader.read('2023-01-01', '2023-01-31', timeframe='1h')  # doctest: +SKIP
            >>> print(f"VWAP range: {hourly['vwap'].min():.2f} - {hourly['vwap'].max():.2f}")  # doctest: +SKIP
            VWAP range: 16420.50 - 17890.75
            >>>
            >>> # Daily bars for trend analysis
            >>> daily = reader.read('2023-01-01', '2023-12-31', timeframe='1D')  # doctest: +SKIP
            >>> daily_returns = daily['close'].pct_change()  # doctest: +SKIP
        """
        # Normalize input parameters
        if isinstance(start_time, str):
            start_time = pd.Timestamp(start_time)
        if isinstance(end_time, str):
            end_time = pd.Timestamp(end_time)

        # If end_time is provided as a date without time, set it to the end of that day
        # to make the range inclusive of the entire end date
        original_end_time = None
        if end_time is not None and end_time.time() == dt.time(0, 0):
            original_end_time = end_time  # Store for later use in resampling
            end_time = end_time + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)

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

        # If we're resampling to a daily timeframe and we have an incomplete last day,
        # we need to take special care
        if timeframe.upper().endswith('D') and original_end_time is not None:
            # Resample to the requested timeframe
            resampled = self._resample(df, timeframe)

            # For daily bars, if the end date is specified without time component,
            # exclude the last day if it's incomplete
            last_valid_day = original_end_time - pd.Timedelta(days=1)
            resampled = resampled[resampled.index <= last_valid_day]
            return resampled
        else:
            # Resample to the requested timeframe normally
            return self._resample(df, timeframe)

    def _resample(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        r"""Apply mathematically correct resampling aggregation to transform 1-second bars into target timeframe.

        This internal method implements the core resampling logic, ensuring that volume-weighted metrics
        are properly recalculated and that statistical properties are preserved across time aggregation.

        **Aggregation Rules Applied:**

        - **OHLC**: Uses first/max/min/last semantics appropriate for price series
        - **Volume & Trades**: Simple summation across the resampling period
        - **VWAP**: Volume-weighted recalculation maintaining accuracy across aggregation
        - **Median Trade Size**: Volume-weighted median computation from per-second medians

        :param df: DataFrame containing 1-second time bars with standard OHLCV columns.
        :param timeframe: Pandas offset string specifying target timeframe (e.g., '5min', '1h', '1D').
        :returns: Resampled DataFrame with aggregated bars at the requested timeframe.
            Periods with no trading activity are automatically excluded.

        .. note::
            The volume-weighted median calculation uses numpy's searchsorted for efficient
            percentile computation, making it suitable for high-frequency resampling operations.
        """
        # --- one grouper reused everywhere ---------------------------------
        grouper = df.index.floor(timeframe)  # fast vectorised

        # ---------- OHLCV & trade-count aggregation -----------------------
        resampled = df.groupby(grouper, sort=False).agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
            "trades": "sum",
        })

        # ---------------------------- VWAP --------------------------------
        vol_sum = df["volume"].groupby(grouper, sort=False).sum()
        vwap = (df["vwap"] * df["volume"]).groupby(grouper, sort=False).sum() / vol_sum
        resampled["vwap"] = vwap.astype("float32")

        # ---------- volume-weighted median of per-second medians ----------
        def w_median(sub: pd.DataFrame) -> float:
            if sub.empty:
                return np.nan
            sort_idx = np.argsort(sub["median_trade_size"].values)
            sizes = sub["median_trade_size"].values[sort_idx]
            weights = sub["trades"].values[sort_idx].astype(np.float64)

            cum_w = np.cumsum(weights)
            cutoff = cum_w[-1] * 0.5
            return float(sizes[np.searchsorted(cum_w, cutoff, side="left")])

        resampled["median_trade_size"] = (
            df.groupby(grouper, sort=False, observed=True).apply(w_median).astype("float32")
        )

        # ---------- final cleanup -----------------------------------------
        # Drop periods with no trades (NaN open). Keeps index monotone.
        resampled = resampled.dropna(subset=["open"])

        return resampled
