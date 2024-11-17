import numpy as np
import pandas as pd
from numba import njit
from finmlkit.bars.base import FootprintData
from typing import Tuple

# TODO: Rewrite to numpy typy documetation
# ------------------------------------
# Volume Profile Calculations
# ------------------------------------
class VolumePro:
    """
    Encapsulates numba functions for smoother calling and parameter setting.
    """
    def __init__(self, window_size_sec=1800, bin_size=5, va_pct=68.34):
        """
        Initialize the Volume Profile calculator with the given parameters.

        :param window_size_sec: Size of the rolling window in seconds
        :param bin_size: Size of the price level bins
        :param va_pct: Value area percentage
        """
        self.window_size_sec = window_size_sec
        self.bin_size = bin_size
        self.va_pct = va_pct

    def reset_parameters(self, window_size_sec=None, bin_size=None, va_pct=None):
        """
        Reset the parameters of the Volume Profile calculator.
        :param window_size_sec: Size of the rolling window in seconds
        :param bin_size: Size of the price level bins
        :param va_pct: Value area percentage
        """
        self.window_size_sec = window_size_sec if window_size_sec is not None else self.window_size_sec
        self.bin_size = bin_size if bin_size is not None else self.bin_size
        self.va_pct = va_pct if va_pct is not None else self.va_pct

    def compute(self, bars: pd.DataFrame, fp_data: FootprintData) -> Tuple[np.array, np.array, np.array]:
        """
        Compute the volume profile parameters (POC, HVA, LVA) in a rolling window fashion.
        :param bars: dynamic bars dataframe (lows and highs needed for the calculations)
        :param fp_data: footprint data container
        :return: POC prices, HVA prices, LVA prices
        """
        # assert that bars length and footprint data length are the same
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

    def compute_range(self, bars: pd.DataFrame, fp_data: FootprintData, start, end) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute the volume profile parameters (POC, HVA, LVA) in a rolling window fashion for a specified range.

        :param bars: dynamic bars dataframe (lows and highs needed for the calculations)
        :param fp_data: footprint data container
        :param start: start timestamp for the aggregation in any format (e.g. datetime string, timestamp)
        :param end: end timestamp for the aggregation in any format (e.g. datetime string, timestamp)

        :return: bar timestamps, POC prices, HVA prices, LVA prices (on the range)
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

        # Convert bar_timestamps to DatetimeIndex to match bars index format
        fp_datetime_index = pd.to_datetime(fp_data_subset.bar_timestamps, unit='ns')

        # Align bars with the footprint data using the datetime index
        bars_subset = bars.loc[fp_datetime_index]

        # Call the compute method for the subset
        poc_prices, hva_prices, lva_prices = self.compute(bars_subset, fp_data_subset)

        # Return the corresponding timestamps too
        return fp_data_subset.bar_timestamps, poc_prices, hva_prices, lva_prices


@njit(nogil=True)
def aggregate_footprint(ts: np.array, highs: np.array, lows: np.array, price_levels: list[np.array],
                        buy_volumes: list[np.array], sell_volumes: list[np.array], start_ts: int, end_ts: int,
                        price_tick: float):
    """
    Aggregate the volume footprint data on the specified time window to calculate the POC, HVA, and LVA.
    :param ts: (1d np.array int64) bar timestamps
    :param highs: (1d np.array float64) high prices for each bar timestamp
    :param lows: (1d np.array float64) low prices for each bar timestamp
    :param price_levels: list of price levels (1d numpy arrays)
    :param buy_volumes: list of corresponding buy volume (1d numpy arrays)
    :param sell_volumes: list of corresponding sell volume (1d numpy arrays)
    :param start_ts: start timestamp for the aggregation
    :param end_ts: end timestamp for the aggregation
    :param price_tick: price tick size

    :return: price_levels (1D numpy array), total_buy_volumes (1D numpy array), total_sell_volumes (1D numpy array)
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
def bucket_price_levels(all_price_levels: np.array, total_volumes: np.array, bin_size: int):
    """
    Bucket the price levels into larger bins together with their corresponding volumes to decrease noise.

    :param all_price_levels: 1d array of all price levels
    :param total_volumes: 1d array of total volumes corresponding to the price levels
    :param bin_size: the fixed size of each bin (should be an odd number)

    :return: binned_price_levels, binned_volumes
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
def comp_poc_hva_lva(price_levels, volumes, va_pct=68.34):
    """
    Calculate the POC (point of control), HVA (high value area), and LVA (low value area) for the given price levels and volumes.

    :param price_levels: price levels in ascending order
    :param volumes: volumes at each price level
    :param va_pct: volume area percentage (default is 68.34)

    :return: POC, HVA, LVA
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
def volume_profile_rolling(ts: np.array, highs: np.array, lows: np.array,
                           price_levels: list[np.array], buy_volumes: list[np.array],
                           sell_volumes: list[np.array],
                           window_size_sec: float, bin_size: int = None, price_tick: float = None,
                           va_pct: float = 68.34):
    """
    Compute the volume profile parameters (POC, HVA, LVA) in a rolling window fashion.
    Example usage:
    poc_prices, hva_prices, lva_prices =
    compute_volume_profile(dynabars.index.values.astype(np.int64),
                           fp_data[1], fp_data[2], fp_data[3],
                           window_size_sec=1800, bin_size=5, price_tick=0.1)

    :param ts: timestamps of the bars in nanoseconds
    :param highs: high prices for each bar timestamp
    :param lows: low prices for each bar timestamp
    :param price_levels: price levels for each bar timestamp (output of the footprint function)
    :param buy_volumes: buy volumes corresponding for each price level (output of the footprint function)
    :param sell_volumes: sell volumes corresponding for each price level (output of the footprint function)
    :param window_size_sec: rolling window size in seconds on which the volume profile is calculated
    :param bin_size: the bin size for bucketing the price levels together (should be an odd number) (optional! default is None)
    :param price_tick: the minimum price tick size (default is 0.1 for ETH and BTC) for bucketing the price levels
    :param va_pct: volume area percentage (default is 68.34)

    :return: poc_price (1d array), hva_price (1d array), lva_price (1d array) for each bar timestamp
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
def trim_trailing_zeros(price_levels: np.array, volumes: np.array):
    """
    Trim the trailing zeros from the price levels and volumes arrays.
    :param price_levels: 1D array of price levels
    :param volumes: 1D array of volumes
    :return: trimmed_price_levels, trimmed_volumes
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
def volume_profile_developing(ts: np.array, highs: np.array, lows: np.array,
                              price_levels: list[np.array], buy_volumes: list[np.array], sell_volumes: list[np.array],
                              start_ts: int, end_ts: int, bin_size: int = None, price_tick: float = None,
                              va_pct: float = 68.34):
    """
    Compute the volume profile parameters (POC, HVA, LVA) in a developing fashion between the start and end timestamps.

    :param ts: timestamps of the bars in nanoseconds
    :param price_levels: price levels for each bar timestamp (output of the footprint function)
    :param buy_volumes: buy volumes corresponding for each price level (output of the footprint function)
    :param sell_volumes: sell volumes corresponding for each price level (output of the footprint function)
    :param start_ts: start nanoseconds timestamp for the aggregation
    :param end_ts: end nanoseconds timestamp for the aggregation
    :param bin_size: the bin size for bucketing the price levels together (should be an odd number) (optional! default is None)
    :param price_tick: the minimum price tick size (default is 0.1 for ETH and BTC) for bucketing the price levels
    :param va_pct: volume area percentage (default is 68.34)

    :return: poc_price (1d array), hva_price (1d array), lva_price (1d array) for each bar timestamp
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
