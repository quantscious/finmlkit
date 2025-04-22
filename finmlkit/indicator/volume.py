import numpy as np
import pandas as pd
from numba import njit
from finmlkit.bar.data_model import FootprintData
from typing import Union


class VolumePro:
    """
    Encapsulates numba functions for smoother calling and parameter setting.
    """

    def __init__(self, window_size_sec: int=1800, bin_size: int=5, va_pct: float=68.34):
        """
        Initialize the Volume Profile calculator with the given parameters.

        :param window_size_sec: Size of the rolling window in seconds.
        :param bin_size: Size of the price level bins.
        :param va_pct: Value area percentage.
        :note:
            This sets the rolling window size, the bin size for price level bucketing, and the value area percentage
            used for determining the high and low value areas (HVA and LVA).
            Default values are window_size_sec=1800, bin_size=5, va_pct=68.34.
        """
        self.window_size_sec = window_size_sec
        self.bin_size = bin_size
        self.va_pct = va_pct

    def reset_parameters(self, window_size_sec: int=None, bin_size: int=None, va_pct: float=None):
        """
        Reset the parameters of the Volume Profile calculator.

        :param window_size_sec: Optional new window size in seconds. If None, the existing value is retained.
        :param bin_size: Optional new bin size for price level bucketing. If None, the existing value is retained.
        :param va_pct: Optional new value area percentage. If None, the existing value is retained.
        :note:
            This method allows dynamic reconfiguration of the rolling window size, price bin size, or value area percentage
            for the Volume Profile calculations. Any parameter left as None will retain its prior value.
        """
        self.window_size_sec = window_size_sec if window_size_sec is not None else self.window_size_sec
        self.bin_size = bin_size if bin_size is not None else self.bin_size
        self.va_pct = va_pct if va_pct is not None else self.va_pct

    def compute(self, bars: pd.DataFrame, fp_data: FootprintData) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute the volume profile parameters (POC, HVA, LVA) in a rolling window fashion.

        :param bars: DataFrame containing dynamic bars (must include `high` and `low` columns).
        :param fp_data: FootprintData object with price levels and volume information.
        :returns: Tuple of POC, HVA, and LVA prices (as NumPy arrays).
        :raises AssertionError: If `bars` and `fp_data` have different lengths.
        :note:
            The computation is performed in a rolling window fashion, using the set window size, bin size, and value area percentage.
            The bars DataFrame must contain 'high' and 'low' columns, and the footprint data must be aligned in length.
            Returned arrays represent the Point of Control (POC), High Value Area (HVA), and Low Value Area (LVA) prices for each bar.
            The computation replaces starting zeros with NaN to indicate insufficient data at the window start.
        """
        # assert that bar length and footprint data length are the same
        assert len(bars) == len(fp_data.bar_timestamps), "Bars and footprint data should have the same length."

        # Cast the footprint data to a numba list for numba calculations
        fp_data.cast_to_numba_list()

        poc_prices, hva_prices, lva_prices = volume_profile_rolling(
            fp_data.bar_timestamps, bars.high.values, bars.low.values,
            fp_data.price_levels, fp_data.buy_volumes, fp_data.sell_volumes,
            window_size_sec=self.window_size_sec, bin_size=self.bin_size, price_tick=fp_data.price_tick,
            va_pct=self.va_pct
        )

        # Convert the prices to actual price units
        poc_prices = poc_prices * fp_data.price_tick
        hva_prices = hva_prices * fp_data.price_tick
        lva_prices = lva_prices * fp_data.price_tick

        # Replace starting zeros with NaNs
        poc_prices = np.where(poc_prices == 0, np.nan, poc_prices)
        hva_prices = np.where(hva_prices == 0, np.nan, hva_prices)
        lva_prices = np.where(lva_prices == 0, np.nan, lva_prices)

        return poc_prices, hva_prices, lva_prices

    def compute_range(self, bars: pd.DataFrame, fp_data: FootprintData,
                      start: Union[str, int, pd.Timestamp],
                      end: Union[str, int, pd.Timestamp]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute the volume profile (POC, HVA, LVA) in a rolling window fashion for a given time range.

        :param bars: DataFrame containing dynamic bars with `high` and `low`.
        :param fp_data: FootprintData object containing volume profiles.
        :param start: Start timestamp for slicing (str, int, or pd.Timestamp).
        :param end: End timestamp for slicing (same type as `start`).
        :returns: Tuple of bar timestamps, POC, HVA, and LVA prices.
        :raises AssertionError: If `bars` and `fp_data` lengths differ or `start` and `end` are of different types.
        :note:
            This method computes the rolling window volume profile (POC, HVA, LVA) for a specified time range.
            The range is set by the `start` and `end` timestamps, which must be of the same type.
            The method internally adjusts the start timestamp for window warm-up and aligns bar and footprint data by timestamp.
            Returns arrays for the bar timestamps and the computed POC, HVA, and LVA prices over the specified range.
        """
        assert len(bars) == len(fp_data.bar_timestamps), "Bars and footprint data should have the same length."
        assert type(start) is type(end), "Start and end should be of the same type."

        # Convert start and end to pandas Timestamps
        if isinstance(start, int):
            end = pd.to_datetime(end)  # If it is str we want to keep it as str for pandas slice locs

        start = pd.to_datetime(start)
        # Subtract the window size from the start timestamp to eliminate the first window warm-up
        adjusted_start = start - pd.Timedelta(seconds=self.window_size_sec)

        # Use FootprintBT's slicing to get the range
        fp_data_subset = fp_data[adjusted_start:end]

        # Convert bar_timestamps to DatetimeIndex to match bar index format
        fp_datetime_index = pd.to_datetime(fp_data_subset.bar_timestamps, unit='ns')

        # Align bar with the footprint data using the datetime index
        bars_subset = bars.loc[fp_datetime_index]

        # Call the compute method for the subset
        poc_prices, hva_prices, lva_prices = self.compute(bars_subset, fp_data_subset)

        # Return the corresponding timestamps too
        return fp_data_subset.bar_timestamps, poc_prices, hva_prices, lva_prices


@njit(nogil=True)
def aggregate_footprint(ts: np.ndarray,
                        highs: np.ndarray, lows: np.ndarray,
                        price_levels: list[np.ndarray],
                        buy_volumes: list[np.ndarray], sell_volumes: list[np.ndarray],
                        start_ts: int, end_ts: int,
                        price_tick: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Aggregate the volume footprint data within a specified time window to calculate the POC, HVA, and LVA.

    :param ts: Array of timestamps (int64) for each bar in nanoseconds.
    :param highs: High prices for each bar.
    :param lows: Low prices for each bar.
    :param price_levels: List of arrays of price levels for each bar.
    :param buy_volumes: List of arrays of buy volumes for each price level.
    :param sell_volumes: List of arrays of sell volumes for each price level.
    :param start_ts: Start timestamp of the aggregation window.
    :param end_ts: End timestamp of the aggregation window.
    :param price_tick: Tick size used to discretize prices.
    :returns: Tuple of:
        - complete_price_levels: Aggregated price levels in integer tick units.
        - aligned_buy_volumes: Aggregated buy volumes aligned to tick grid.
        - aligned_sell_volumes: Aggregated sell volumes aligned to tick grid.
    :raises AssertionError: If input lists/arrays are not aligned in length or empty.
    """
    assert len(ts) == len(highs) == len(lows) == len(price_levels) == len(buy_volumes) == len(
        sell_volumes) > 0, "Input arrays should have the same length and be non-empty."

    # Find the indices for the start and end timestamps indices for the aggregation
    start_idx = np.searchsorted(ts, start_ts)
    end_idx = np.searchsorted(ts, end_ts, side='right')

    # If start_idx is equal to end_idx, set start_idx to the previous index
    if start_idx == end_idx:
        start_idx = max(0, start_idx - 1)

    # Select current data windows
    high_window = highs[start_idx:end_idx]
    low_window = lows[start_idx:end_idx]

    price_levels_window = price_levels[start_idx:end_idx]
    buy_volumes_window = buy_volumes[start_idx:end_idx]
    sell_volumes_window = sell_volumes[start_idx:end_idx]
    n_bars_window = len(price_levels_window)

    # Get minimum and maximum price levels for the current window
    min_price_level, max_price_level = np.min(low_window), np.max(high_window)

    # Convert price to price_tick units
    min_price_level = int(round(min_price_level / price_tick))
    max_price_level = int(round(max_price_level / price_tick))

    # Generate complete price levels in steps and convert back to actual price levels
    complete_price_levels = np.arange(min_price_level, max_price_level + 1, dtype=np.int32)
    n_price_levels_window = len(complete_price_levels)
    # -----------------------------------------------------------------------------------

    # Initialize aligned volumes with zeros
    aligned_buy_volumes = np.zeros(n_price_levels_window, dtype=np.float32)
    aligned_sell_volumes = np.zeros(n_price_levels_window, dtype=np.float32)

    # Align the volumes with the unified price levels for each bar timestamp
    for t in range(n_bars_window):
        # Find the indices to place the volumes into the aligned arrays
        indices = np.searchsorted(complete_price_levels, price_levels_window[t])

        # Cumulate the volumes in the aligned arrays
        aligned_buy_volumes[indices] += buy_volumes_window[t]
        aligned_sell_volumes[indices] += sell_volumes_window[t]

    return complete_price_levels, aligned_buy_volumes, aligned_sell_volumes


@njit(nogil=True)
def bucket_price_levels(all_price_levels: np.ndarray, total_volumes: np.ndarray, bin_size: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Bucket the price levels and associated volumes using a fixed-size bin to reduce noise.

    :param all_price_levels: Array of all price levels.
    :param total_volumes: Corresponding total volumes.
    :param bin_size: Width of each bin (should be odd).
    :returns: Tuple of:
        - binned_price_levels: Midpoints of each bucket.
        - binned_volumes: Aggregated volumes per bucket.
    :raises AssertionError: If input arrays are empty or of mismatched length.
    :raises ValueError: If a price level falls outside defined bins.
    """
    assert bin_size >= 3, "Bin size should be larger than 3"
    assert len(all_price_levels) == len(
        total_volumes) > 0, "Price levels and volumes should have the same length and non-empty"

    # Ensure bin_size is odd
    if bin_size % 2 == 0:
        bin_size += 1

    # Define the bin edges
    min_price = np.min(all_price_levels)  # integer in price_tick units
    max_price = np.max(all_price_levels)  # integer in price_tick units
    bin_edges = np.arange(min_price, max_price + bin_size, bin_size, dtype=np.int32)
    n_bins = len(bin_edges) - 1

    # Ensure that there is at least one bin
    if len(bin_edges) < 2:
        bin_edges = np.array([min_price, max_price + 1], dtype=np.int32)
        print("WARNING! There is only one bin at price level bucketing...")

    # Digitize the price levels into price level buckets
    bin_indices = np.digitize(all_price_levels, bin_edges) - 1

    # Check if there are any leftover price levels (not fitting into the bins)
    has_leftovers = bin_indices[-1] == n_bins

    # Initialize arrays for binned price levels and volumes
    if has_leftovers:
        binned_volumes = np.zeros(n_bins + 1, dtype=np.float32)  # +1 to handle potential leftover bin
        binned_price_levels = np.zeros(n_bins + 1, dtype=np.int32)
    else:
        binned_volumes = np.zeros(n_bins, dtype=np.float32)
        binned_price_levels = np.zeros(n_bins, dtype=np.int32)

    # Calculate the midpoint of each bin to represent the binned price levels using vectorized computation
    binned_price_levels[:n_bins] = ((bin_edges[:-1] + bin_edges[1:] - 1) // 2).astype(np.int32)
    if has_leftovers:
        binned_price_levels[n_bins] = max_price  # Add an entry for the potential leftover bin

    # Aggregate volumes into bins
    for i in range(len(all_price_levels)):
        bin_idx = bin_indices[i]
        if 0 <= bin_idx < n_bins:
            # We found the corresponding bucket for the price level
            binned_volumes[bin_idx] += total_volumes[i]
        elif bin_idx == n_bins:
            # We have a leftover price level
            binned_volumes[n_bins] += total_volumes[i]
        else:
            raise ValueError("BUG: Bin index is out of bounds while bucketing price levels...")

    return binned_price_levels, binned_volumes


@njit(nogil=True)
def comp_poc_hva_lva(price_levels: np.ndarray, volumes: np.ndarray, va_pct=68.34) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate the POC (Point of Control), HVA (High Value Area), and LVA (Low Value Area)
    for a given volume profile.

    :param price_levels: Price levels (sorted ascending).
    :param volumes: Corresponding volumes.
    :param va_pct: Value area percentage (default 68.34).
    :returns: Tuple of:
        - poc_price: Price with the highest volume.
        - hva_price: High value area bound.
        - lva_price: Low value area bound.
    :raises AssertionError: If inputs are empty or mismatched in length.
    """
    assert len(price_levels) == len(volumes) > 0, "Price levels and volumes must have the same length and be non-empty."

    n_levels = len(price_levels)
    total_volume = np.sum(volumes)
    poc_index = np.argmax(volumes)
    poc_price = price_levels[poc_index]
    va_thrs = total_volume * (va_pct / 100.0)

    cum_volume = volumes[poc_index]
    hva_price = poc_price
    lva_price = poc_price

    up_idx = poc_index + 1
    down_idx = poc_index - 1

    current_up_volume = 0.0
    current_down_volume = 0.0

    if up_idx < n_levels:
        current_up_volume = volumes[up_idx]
        if up_idx + 1 < n_levels:
            current_up_volume += volumes[up_idx + 1]

    if down_idx >= 0:
        current_down_volume = volumes[down_idx]
        if down_idx - 1 >= 0:
            current_down_volume += volumes[down_idx - 1]

    # Repeat the process until the cumulative volume exceeds the value area threshold
    while cum_volume < va_thrs:
        # Determine the direction of the next move
        if current_up_volume > current_down_volume:
            cum_volume += current_up_volume
            hva_price = price_levels[min(up_idx + 1, n_levels - 1)]
            up_idx += 2
            # Update current_up_volume for the new up_idx
            current_up_volume = -1.0
            if up_idx < n_levels:
                current_up_volume = volumes[up_idx]
                if up_idx + 1 < n_levels:
                    current_up_volume += volumes[up_idx + 1]
        elif current_up_volume < current_down_volume:
            cum_volume += current_down_volume
            lva_price = price_levels[max(down_idx - 1, 0)]
            down_idx -= 2
            # Update current_down_volume for the new down_idx
            current_down_volume = -1.0
            if down_idx >= 0:
                current_down_volume = volumes[down_idx]
                if down_idx - 1 >= 0:
                    current_down_volume += volumes[down_idx - 1]
        elif current_up_volume == current_down_volume != -1.0:
            cum_volume += current_up_volume + current_down_volume
            hva_price = price_levels[min(up_idx + 1, n_levels - 1)]
            lva_price = price_levels[max(down_idx - 1, 0)]
            up_idx += 2
            down_idx -= 2
            # Update both current_up_volume and current_down_volume
            current_up_volume = -1.0
            if up_idx < n_levels:
                current_up_volume = volumes[up_idx]
                if up_idx + 1 < n_levels:
                    current_up_volume += volumes[up_idx + 1]
            current_down_volume = -1.0
            if down_idx >= 0:
                current_down_volume = volumes[down_idx]
                if down_idx - 1 >= 0:
                    current_down_volume += volumes[down_idx - 1]
        else:
            print("BUG: Stuck in loop while calculating POC, HVA, and LVA.")
            break

    return poc_price, hva_price, lva_price


@njit(nogil=True)
def volume_profile_rolling(ts: np.ndarray, highs: np.ndarray, lows: np.ndarray,
                           price_levels: list[np.ndarray], buy_volumes: list[np.ndarray],
                           sell_volumes: list[np.ndarray],
                           window_size_sec: float, bin_size: int = None, price_tick: float = None,
                           va_pct: float = 68.34) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute rolling volume profiles over a fixed-width time window.

    :param ts: Nanosecond timestamps per bar.
    :param highs: High prices per bar.
    :param lows: Low prices per bar.
    :param price_levels: List of price levels per bar.
    :param buy_volumes: List of buy volumes per bar.
    :param sell_volumes: List of sell volumes per bar.
    :param window_size_sec: Width of the rolling time window.
    :param bin_size: Optional price level binning width.
    :param price_tick: Price tick size for discretization.
    :param va_pct: Value area percentage.
    :returns: Tuple of POC, HVA, LVA price series aligned to input bars.
    :raises AssertionError: If input arrays are empty or misaligned in length.
    """
    assert len(ts) == len(highs) == len(lows) == len(price_levels) == len(buy_volumes) == len(sell_volumes) > 0, "Input arrays should have the same length and be non-empty."

    n_bars = len(price_levels)
    poc_prices = np.zeros(n_bars, dtype=np.int32)
    hva_prices = np.zeros(n_bars, dtype=np.int32)
    lva_prices = np.zeros(n_bars, dtype=np.int32)

    window_interval_ns = int(window_size_sec * 1e9)
    first_interval_idx = np.searchsorted(ts, ts[0] + window_interval_ns)

    for i in range(first_interval_idx, n_bars):
        end_ts = ts[i]
        start_ts = end_ts - window_interval_ns

        all_price_levels, total_buy_volumes, total_sell_volumes = aggregate_footprint(ts, highs, lows,
                                                                                      price_levels,
                                                                                      buy_volumes,
                                                                                      sell_volumes,
                                                                                      start_ts, end_ts,
                                                                                      price_tick)

        total_volumes = total_buy_volumes + total_sell_volumes

        # TODO: Further volume profile calculations: 1.) Volume+Delta 2.) Ticks+Delta 3.) Min/max 4.) Imbalance

        # Bucket price levels
        if bin_size is not None:
            assert price_tick is not None, "Price tick should be provided for bucketing price levels correctly..."
            all_price_levels, total_volumes = bucket_price_levels(all_price_levels, total_volumes, bin_size)

        # Calculate the POC, HVA, and LVA
        poc_price, hva_price, lva_price = comp_poc_hva_lva(all_price_levels, total_volumes, va_pct)

        poc_prices[i] = poc_price
        hva_prices[i] = hva_price
        lva_prices[i] = lva_price

    return poc_prices, hva_prices, lva_prices


# ---------------------------------------
# Developing Volume Profile Calculations
# ---------------------------------------
@njit(nogil=True)
def trim_trailing_zeros(price_levels: np.ndarray, volumes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Trim trailing zero volumes from a price-level volume profile.

    :param price_levels: Array of price levels.
    :param volumes: Corresponding volumes.
    :returns: Trimmed price levels and volumes.
    """
    # round aggregated volumes to avoid floating point errors
    volumes = np.round(volumes, 8)

    left_idx = 0
    right_idx = len(price_levels) - 1

    # Locate zeros from the left
    while left_idx <= right_idx and volumes[left_idx] == 0:
        left_idx += 1

    # Locate zeros from the right
    while right_idx >= left_idx and volumes[right_idx] == 0.0:
        right_idx -= 1

    # Trim the arrays
    price_levels = price_levels[left_idx:right_idx + 1]
    volumes = volumes[left_idx:right_idx + 1]

    return price_levels, volumes


@njit(nogil=True)
def volume_profile_developing(ts: np.ndarray, highs: np.ndarray, lows: np.ndarray,
                              price_levels: list[np.ndarray], buy_volumes: list[np.ndarray], sell_volumes: list[np.ndarray],
                              start_ts: int, end_ts: int, bin_size: int = None, price_tick: float = None,
                              va_pct: float = 68.34) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute a developing volume profile between two timestamps using cumulative aggregation.

    :param ts: Nanosecond timestamps per bar.
    :param highs: High prices per bar.
    :param lows: Low prices per bar.
    :param price_levels: List of price levels per bar.
    :param buy_volumes: List of buy volumes per bar.
    :param sell_volumes: List of sell volumes per bar.
    :param start_ts: Start time in nanoseconds.
    :param end_ts: End time in nanoseconds.
    :param bin_size: Optional price level bucketing bin width.
    :param price_tick: Tick size for price bucketing.
    :param va_pct: Value area percentage.
    :returns: Tuple of timestamps, POC, HVA, and LVA series for the range.
    :raises AssertionError: If input arrays are empty or misaligned in length.
    """
    assert len(ts) == len(highs) == len(lows) == len(price_levels) == len(buy_volumes) == len(
        sell_volumes) > 0, "Input arrays should have the same length and be non-empty."

    start_idx = np.searchsorted(ts, start_ts)
    end_idx = np.searchsorted(ts, end_ts, side='right')
    n_bars = end_idx - start_idx

    poc_prices = np.zeros(n_bars, dtype=np.int32)
    hva_prices = np.zeros(n_bars, dtype=np.int32)
    lva_prices = np.zeros(n_bars, dtype=np.int32)

    # Calculate the maximum price level range for the window
    min_price_level = np.min(lows[start_idx:end_idx])
    max_price_level = np.max(highs[start_idx:end_idx])

    # Convert to price tick units
    min_price_level = int(round(min_price_level / price_tick))
    max_price_level = int(round(max_price_level / price_tick))

    # Cumulate the footprint data
    price_levels_window = np.arange(min_price_level, max_price_level + 1, dtype=np.int32)
    developing_buy_volumes = np.zeros_like(price_levels_window, dtype=np.float32)
    developing_sell_volumes = np.zeros_like(price_levels_window, dtype=np.float32)

    for t in range(start_idx, end_idx):
        # Aggregate the footprint data
        indices = np.searchsorted(price_levels_window, price_levels[t])
        developing_buy_volumes[indices] += buy_volumes[t]
        developing_sell_volumes[indices] += sell_volumes[t]

        developing_total_volumes = developing_buy_volumes + developing_sell_volumes

        # Bucket price levels
        if bin_size is not None:
            assert price_tick is not None, "Price tick should be provided for bucketing price levels correctly..."

            # Before bucketing, we have to trim the trailing zeros
            current_price_levels, developing_total_volumes = trim_trailing_zeros(price_levels_window,
                                                                                 developing_total_volumes)
            # Bucket the price levels and volumes
            bucketed_price_levels, bucketed_total_volumes = bucket_price_levels(current_price_levels,
                                                                                developing_total_volumes, bin_size)

            # Calculate the POC, HVA, and LVA
            poc_price, hva_price, lva_price = comp_poc_hva_lva(bucketed_price_levels, bucketed_total_volumes,
                                                               va_pct)
        else:
            # Calculate the POC, HVA, and LVA
            poc_price, hva_price, lva_price = comp_poc_hva_lva(price_levels_window, developing_total_volumes,
                                                               va_pct)

        poc_prices[t] = poc_price
        hva_prices[t] = hva_price
        lva_prices[t] = lva_price

    return ts[start_idx:end_idx], poc_prices, hva_prices, lva_prices
