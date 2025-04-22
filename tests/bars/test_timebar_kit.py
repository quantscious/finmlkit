import numpy as np
import pytest
import os
os.environ['NUMBA_DISABLE_JIT'] = '1'  # Disable JIT for testing (we can debug numba functions this way)

from finmlkit.bar.kit import TimeBarKit

def test_time_bar_kit():
    import pandas as pd

    path = "../test_resources/BTCUSDT-trades-2025-03.csv"
    df = pd.read_csv(path)

    # Preprocess
    tbk = TimeBarKit(df, 60)
    klines = tbk.build_ohlcv()

    # Pandas implementation
    dfp = tbk._raw_data
    dfp['timestamp'] = pd.to_datetime(dfp['timestamp'])

    # Set 'timestamp' as the index
    dfp.set_index('timestamp', inplace=True)

    # Resample to 1-minute intervals and calculate OHLCV
    ohlcv = dfp.resample('1T').agg({
        'price': ['first', 'max', 'min', 'last'],  # Open, High, Low, Close
        'amount': 'sum'  # Volume
    })

    # Rename columns for clarity
    ohlcv.columns = ['open', 'high', 'low', 'close', 'volume']

    # make them same length (our method strictly keeps only full periods)
    ohlcv = ohlcv.iloc[1:-1]

    # Check if the lengths are the same
    assert len(klines) == len(ohlcv), "Lengths of klines and ohlcv do not match"
    # Check if the values are the same
    ohlcv.index.freq = None
    pd.testing.assert_frame_equal(klines[ohlcv.columns], ohlcv, check_dtype=False)
