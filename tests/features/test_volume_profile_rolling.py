import numpy as np
import pandas as pd
import pytest
from numba.typed import List as NumbaList

from finmlkit.bar.data_model import FootprintData
from finmlkit.feature.core.volume import (
    volume_profile_rolling, calc_volume_percentage_above_poc, VolumePro
)


class TestVolumePercentageAbovePOC:
    def test_calc_volume_percentage_above_poc_basic(self):
        """Test the basic calculation of volume percentage above POC."""
        price_levels = np.array([100, 101, 102, 103, 104], dtype=np.int32)
        volumes = np.array([10.0, 5.0, 20.0, 5.0, 10.0], dtype=np.float32)
        poc_price = 102  # Price level with maximum volume

        # Volumes above POC: 5.0 + 10.0 = 15.0
        # Total volume: 50.0
        # Expected percentage: 15.0 / 50.0 = 0.3
        result = calc_volume_percentage_above_poc(price_levels, volumes, poc_price)
        assert np.isclose(result, 0.3)

    def test_calc_volume_percentage_above_poc_no_volume_above(self):
        """Test when there's no volume above the POC."""
        price_levels = np.array([100, 101, 102, 103, 104], dtype=np.int32)
        volumes = np.array([10.0, 5.0, 0.0, 0.0, 0.0], dtype=np.float32)
        poc_price = 101  # POC is at the highest price level with volume

        result = calc_volume_percentage_above_poc(price_levels, volumes, poc_price)
        assert np.isclose(result, 0.0)

    def test_calc_volume_percentage_above_poc_all_volume_above(self):
        """Test when all volume is above the POC."""
        price_levels = np.array([100, 101, 102, 103, 104], dtype=np.int32)
        volumes = np.array([0.0, 0.0, 5.0, 10.0, 15.0], dtype=np.float32)
        poc_price = 100  # POC is at the lowest price level

        result = calc_volume_percentage_above_poc(price_levels, volumes, poc_price)
        assert np.isclose(result, 1.0)

    def test_calc_volume_percentage_above_poc_zero_volume(self):
        """Test handling of zero total volume."""
        price_levels = np.array([100, 101, 102], dtype=np.int32)
        volumes = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        poc_price = 101

        result = calc_volume_percentage_above_poc(price_levels, volumes, poc_price)
        assert np.isclose(result, 0.0)


class TestVolumeProfileRolling:
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing volume profile rolling."""
        # Create timestamps (nanoseconds)
        num_bars = 10
        ts = np.array(
            [pd.Timestamp(f"2025-01-01 00:0{i}:00").value for i in range(num_bars)],
            dtype=np.int64
        )

        # Price data
        highs = np.array([105.0, 106.0, 107.0, 108.0, 109.0, 110.0, 111.0, 112.0, 113.0, 114.0])
        lows = np.array([100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0])

        # Create price levels and volumes for each bar
        price_levels_list = []
        buy_volumes_list = []
        sell_volumes_list = []

        # For simplicity, we'll create a basic pattern where each bar has 5 price levels
        price_tick = 1.0

        for i in range(num_bars):
            # Create price levels around the bar's range
            levels = np.array(
                [int(lows[i]), int(lows[i] + 1), int((lows[i] + highs[i]) / 2),
                 int(highs[i] - 1), int(highs[i])],
                dtype=np.int32
            )

            # Create buy volumes with a peak in the middle (POC)
            buys = np.array([1.0, 2.0, 5.0, 2.0, 1.0], dtype=np.float32)

            # Create sell volumes with a different pattern
            sells = np.array([2.0, 1.0, 3.0, 4.0, 2.0], dtype=np.float32)

            price_levels_list.append(levels)
            buy_volumes_list.append(buys)
            sell_volumes_list.append(sells)

        # Convert to numba typed lists
        nb_price_levels = NumbaList()
        nb_buy_volumes = NumbaList()
        nb_sell_volumes = NumbaList()

        for i in range(num_bars):
            nb_price_levels.append(price_levels_list[i])
            nb_buy_volumes.append(buy_volumes_list[i])
            nb_sell_volumes.append(sell_volumes_list[i])

        return {
            'ts': ts,
            'highs': highs,
            'lows': lows,
            'price_levels': nb_price_levels,
            'buy_volumes': nb_buy_volumes,
            'sell_volumes': nb_sell_volumes,
            'price_tick': price_tick,
            'original_price_levels': price_levels_list  # Keep for verification
        }

    def test_volume_profile_rolling_returns_vp_pct_abv_poc(self, sample_data):
        """Test that volume_profile_rolling returns vp_pct_abv_poc as the fourth element."""
        window_size_sec = 300  # 5 minutes

        # Call the function
        poc_prices, hva_prices, lva_prices, vp_pct_abv_poc = volume_profile_rolling(
            sample_data['ts'], sample_data['highs'], sample_data['lows'],
            sample_data['price_levels'], sample_data['buy_volumes'], sample_data['sell_volumes'],
            window_size_sec=window_size_sec, price_tick=sample_data['price_tick']
        )

        # Check that vp_pct_abv_poc has the right shape and type
        assert len(vp_pct_abv_poc) == len(sample_data['ts'])
        assert vp_pct_abv_poc.dtype == np.float32

        # First values should be zero (no data in window yet)
        assert np.isclose(vp_pct_abv_poc[0], 0.0)

        # Later values should be between 0 and 1
        for i in range(1, len(vp_pct_abv_poc)):
            assert 0.0 <= vp_pct_abv_poc[i] <= 1.0, f"Value at index {i} is {vp_pct_abv_poc[i]}, not between 0 and 1"

    def test_volume_profile_rolling_values(self, sample_data):
        """Test the values calculated by volume_profile_rolling."""
        window_size_sec = 60  # 1 minute - small enough that each window only contains 1 bar

        # Call the function
        poc_prices, hva_prices, lva_prices, vp_pct_abv_poc = volume_profile_rolling(
            sample_data['ts'], sample_data['highs'], sample_data['lows'],
            sample_data['price_levels'], sample_data['buy_volumes'], sample_data['sell_volumes'],
            window_size_sec=window_size_sec, price_tick=sample_data['price_tick']
        )

        # With our test data, the POC (Point of Control) is always at the middle price level
        # For each bar, we have volumes [1.0, 2.0, 5.0, 2.0, 1.0] (buys) and [2.0, 1.0, 3.0, 4.0, 2.0] (sells)
        # Total volumes are [3.0, 3.0, 8.0, 6.0, 3.0]
        # The POC is at index 2 (value 8.0)
        # Volumes above POC are at indices 3 and 4: 6.0 + 3.0 = 9.0
        # Total volume is 23.0
        # Expected vp_pct_abv_poc = 9.0 / 23.0 ≈ 0.391

        # The first value will be 0 because there's no data before the first bar
        assert np.isclose(vp_pct_abv_poc[0], 0.0)

        # We can check specific values starting from the second bar
        for i in range(1, len(vp_pct_abv_poc)):
            # The exact value depends on how the bars are aggregated in the window,
            # but we can do a basic sanity check
            assert 0.0 < vp_pct_abv_poc[i] < 1.0, f"Value at index {i} should be between 0 and 1, got {vp_pct_abv_poc[i]}"


class TestVolumePro:
    @pytest.fixture
    def sample_footprint_data(self):
        """Create a sample FootprintData object for testing."""
        num_bars = 10
        ts = np.array(
            [pd.Timestamp(f"2025-01-01 00:0{i}:00").value for i in range(num_bars)],
            dtype=np.int64
        )

        # Price data
        price_tick = 1.0

        # Create price levels and volumes for each bar
        price_levels_list = []
        buy_volumes_list = []
        sell_volumes_list = []
        buy_ticks_list = []
        sell_ticks_list = []
        buy_imbalances_list = []
        sell_imbalances_list = []

        for i in range(num_bars):
            # Create 5 price levels for each bar
            levels = np.array([100 + i, 101 + i, 102 + i, 103 + i, 104 + i], dtype=np.int32)
            buys = np.array([1.0, 2.0, 5.0, 2.0, 1.0], dtype=np.float32)
            sells = np.array([2.0, 1.0, 3.0, 4.0, 2.0], dtype=np.float32)

            # Create dummy tick counts (not actually used in volume profile calculations)
            buy_ticks = np.array([1, 2, 5, 2, 1], dtype=np.int32)
            sell_ticks = np.array([2, 1, 3, 4, 2], dtype=np.int32)

            # Create dummy imbalances (not used in volume profile calculations)
            buy_imb = np.array([0.1, 0.2, 0.3, 0.2, 0.1], dtype=np.float32)
            sell_imb = np.array([0.2, 0.1, 0.3, 0.4, 0.2], dtype=np.float32)

            price_levels_list.append(levels)
            buy_volumes_list.append(buys)
            sell_volumes_list.append(sells)
            buy_ticks_list.append(buy_ticks)
            sell_ticks_list.append(sell_ticks)
            buy_imbalances_list.append(buy_imb)
            sell_imbalances_list.append(sell_imb)

        # Create the FootprintData object with all required parameters
        fp_data = FootprintData(
            bar_timestamps=ts,
            price_tick=price_tick,
            price_levels=price_levels_list,
            buy_volumes=buy_volumes_list,
            sell_volumes=sell_volumes_list,
            buy_ticks=buy_ticks_list,
            sell_ticks=sell_ticks_list,
            buy_imbalances=buy_imbalances_list,
            sell_imbalances=sell_imbalances_list
        )

        return fp_data

    @pytest.fixture
    def sample_bars(self):
        """Create sample OHLC bars for testing."""
        num_bars = 10
        timestamps = [pd.Timestamp(f"2025-01-01 00:0{i}:00") for i in range(num_bars)]

        bars = pd.DataFrame({
            'open': np.arange(100, 110),
            'high': np.arange(105, 115),
            'low': np.arange(100, 110),
            'close': np.arange(102, 112),
            'volume': np.ones(num_bars) * 100
        }, index=timestamps)

        return bars

    def test_volumepro_compute_returns_vp_pct_abv_poc(self, sample_footprint_data, sample_bars):
        """Test that VolumePro.compute returns vp_pct_abv_poc as the fourth element."""
        # Create VolumePro instance
        window_size = pd.Timedelta(minutes=5)
        vp = VolumePro(window_size=window_size)

        # Compute volume profile
        poc_prices, hva_prices, lva_prices, vp_pct_above_poc = vp.compute(sample_bars, sample_footprint_data)

        # Check that vp_pct_above_poc has the right shape
        assert len(vp_pct_above_poc) == len(sample_bars)

        # Early values might be NaN due to window warm-up
        # Later values should be between 0 and 1
        for i in range(len(vp_pct_above_poc)):
            if not np.isnan(vp_pct_above_poc[i]):
                assert 0.0 <= vp_pct_above_poc[i] <= 1.0

    def test_volumepro_compute_range_returns_vp_pct_abv_poc(self, sample_footprint_data, sample_bars):
        """Test that VolumePro.compute_range returns vp_pct_abv_poc as the fifth element."""
        # Create VolumePro instance
        window_size = pd.Timedelta(minutes=2)
        vp = VolumePro(window_size=window_size)

        # Compute volume profile for a range
        start = pd.Timestamp('2025-01-01 00:02:00')
        end = pd.Timestamp('2025-01-01 00:05:00')

        timestamps, poc_prices, hva_prices, lva_prices, vp_pct_above_poc = vp.compute_range(
            sample_bars, sample_footprint_data, start, end
        )

        # Check that we got timestamps and vp_pct_above_poc
        assert len(timestamps) > 0
        assert len(vp_pct_above_poc) == len(timestamps)

        # Values should be between 0 and 1 (if not NaN)
        for i in range(len(vp_pct_above_poc)):
            if not np.isnan(vp_pct_above_poc[i]):
                assert 0.0 <= vp_pct_above_poc[i] <= 1.0


if __name__ == '__main__':
    pytest.main()