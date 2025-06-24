from numba import prange, njit
from numpy.typing import NDArray
import numpy as np


@njit(nogil=True, parallel=True)
def comp_slippage(event_indices: NDArray[np.int64],
                  event_side: NDArray[np.int8],
                  timestamps: NDArray[np.int64],
                  prices: NDArray[np.float64],
                  amounts: NDArray[np.float64],
                  sides: NDArray[np.int8],
                  dollar_target: float,
                  latency_ms: int) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """
    Calculate slippage based on the given parameters.

    :param event_indices: Event indices indicating the start of each event in the raw trades data
    :param event_side: Sides of the events (1 for buy, -1 for sell)
    :param timestamps: Timestamps of the raw trades data
    :param prices: Prices in the raw trade data
    :param amounts: Traded amounts
    :param sides: Sides of the trades (1 for buy, -1 for sell)
    :param dollar_target: Market order target in dollars
    :param latency_ms: Execution latency in milliseconds (e.g. latency in ms, eg. 25 ms)
    :return: A tuple of
                - The slippage corresponding to each event.
                - The fill duration for each event in nanoseconds.
    """
    if not len(timestamps) == len(prices) == len(amounts) == len(sides):
        raise ValueError('timestamps and prices must have the same length')
    if len(event_indices) == 0 or len(event_indices) >= len(timestamps):
        raise ValueError('event_indices must be non-empty and not longer than timestamps')
    if len(event_side) != len(event_indices):
        raise ValueError('event_side must have the same length as event_indices')

    n = len(event_indices)
    slippage = np.empty(n, dtype=np.float64)
    slippage.fill(np.nan)
    fill_duration = np.empty(n, dtype=np.int64)

    latency_ns = latency_ms * 1_000_000  # Convert latency from milliseconds to nanoseconds

    for i in prange(n):
        event_idx = event_indices[i]
        side = event_side[i]
        p0 = prices[event_idx]

        start_ts = timestamps[event_idx] + latency_ns
        start_idx = np.searchsorted(timestamps, start_ts, side='left')
        volume, cash = 0.0, 0.0
        idx = start_idx
        while cash < dollar_target and idx < len(timestamps):
            if side == sides[idx]:
                p, a = prices[idx], amounts[idx]
                needed = dollar_target - cash
                take   = min(needed / p, a)
                cash  += p * take
                volume+= take
            idx += 1

        if volume == 0:  # no execution
            slippage[i] = np.nan
            fill_duration[i] = 0
            continue

        vwap = cash / volume
        slippage[i] = side * (vwap - p0) / p0

        last_idx = idx - 1  # make sure we stay inside the array
        if last_idx < start_idx:  # nothing filled
            slippage[i] = np.nan
            fill_duration[i] = 0
            continue
        fill_duration[i] = timestamps[last_idx] - timestamps[start_idx]

    return slippage, fill_duration

