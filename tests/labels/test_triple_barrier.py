import numpy as np
import pytest
import os
# disable jit
#os.environ['NUMBA_DISABLE_JIT'] = "1"

from finmlkit.label.tbm import triple_barrier

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
NS = 1_000_000_000          # seconds → nanoseconds


def _make_timestamps(n):
    """Regular one-second grid in nanoseconds."""
    return (np.arange(n, dtype=np.int64) * NS)


def get_idx_from_ts(timestamps, ts):
    """Find the index corresponding to a given timestamp."""
    return np.searchsorted(timestamps, ts)


# ---------------------------------------------------------------------------
# success-path tests
# ---------------------------------------------------------------------------
def test_upper_barrier_hit():
    """
    Price rises immediately – upper barrier should be hit on the first bar.
    """
    ts = _make_timestamps(5)
    px = np.array([100, 101, 101, 101, 101], dtype=np.float64)
    event_idxs = np.array([0], dtype=np.int64)    # Using index directly
    tgt = np.array([0.005], dtype=np.float64)     # Target for the single event

    label, t_idx, ret, rbr = triple_barrier(
        ts, px, event_idxs, tgt,
        horizontal_barriers=(1.0, 1.0),
        vertical_barrier=3,                        # sec
        side=None,
        min_ret=0.0,                               # Default value, unused in non-meta labeling
    )

    expected_ret = np.log(101 / 100.0)
    assert label[0] == 1
    assert t_idx[0] == 1
    assert ret[0] == pytest.approx(expected_ret)
    assert rbr[0] == 1.0                          # should be exactly 1.0 when barrier is hit


def test_lower_barrier_hit():
    """
    Price drops – lower barrier hit.
    """
    ts = _make_timestamps(5)
    px = np.array([100, 99, 99, 99, 99], dtype=np.float64)
    event_idxs = np.array([0], dtype=np.int64)
    tgt = np.array([0.005], dtype=np.float64)     # Target for the single event

    label, t_idx, ret, rbr = triple_barrier(
        ts, px, event_idxs, tgt,
        horizontal_barriers=(1.0, 1.0),
        vertical_barrier=3,
        side=None,
        min_ret=0.0,                               # Default value, unused in non-meta labeling
    )

    expected_ret = np.log(99 / 100.0)
    assert label[0] == -1
    assert t_idx[0] == 1
    assert ret[0] == pytest.approx(expected_ret)
    assert rbr[0] == 1.0                          # should be exactly 1.0 when barrier is hit


def test_vertical_barrier_only():
    """
    No horizontal barriers ⇒ label comes from the vertical timeout.
    """
    ts = _make_timestamps(5)
    px = np.array([100, 100.2, 100.3, 100.25, 100.2], dtype=np.float64)
    event_idxs = np.array([0], dtype=np.int64)
    tgt = np.array([0.002], dtype=np.float64)      # Target for the single event (0.2%)

    label, t_idx, ret, rbr = triple_barrier(
        ts, px, event_idxs, tgt,
        horizontal_barriers=(np.inf, np.inf),      # disabled
        vertical_barrier=3,
        side=None,
        min_ret=0.0,                               # Default value, unused in non-meta labeling
    )

    assert t_idx[0] == 3                           # 3 s timeout
    assert np.isnan(rbr[0])                        # nan when barriers disabled
    assert label[0] == 1                           # positive return
    assert ret[0] > 0


def test_min_ret_in_meta_labeling():
    """
    Test min_ret parameter in meta-labeling mode (with side information).
    The min_ret parameter only affects meta-labeling output.
    """
    ts = _make_timestamps(5)
    px = np.array([100, 100.2, 100.4, 100.6, 100.8], dtype=np.float64)
    event_idxs = np.array([0], dtype=np.int64)
    tgt = np.array([0.01], dtype=np.float64)  # 1% target
    side = np.array([1], dtype=np.int8)  # Buy side

    # With small min_ret, the return should qualify as a successful trade (label=1)
    label_small, _, ret_small, _ = triple_barrier(
        ts, px, event_idxs, tgt,
        horizontal_barriers=(1.0, 1.0),
        vertical_barrier=4,
        side=side,
        min_ret=0.001,  # Small threshold - 0.1%
    )

    # With large min_ret, the return should be considered unsuccessful (label=0)
    label_large, _, ret_large, _ = triple_barrier(
        ts, px, event_idxs, tgt,
        horizontal_barriers=(1.0, 1.0),
        vertical_barrier=4,
        side=side,
        min_ret=0.01,  # Large threshold - 1% (higher than the actual return)
    )

    # The return should be positive but less than 1%
    assert ret_small[0] > 0
    assert ret_small[0] == ret_large[0]  # Same return in both cases

    # Different labels due to different min_ret thresholds
    assert label_small[0] == 1  # Return > small threshold
    assert label_large[0] == 0  # Return < large threshold


def test_meta_labeling():
    """
    With side information the output label must be (0, 1).
    """
    ts = _make_timestamps(4)
    px = np.array([100, 101, 102, 103], dtype=np.float64)
    event_idxs = np.array([0], dtype=np.int64)
    tgt = np.array([0.004], dtype=np.float64)
    side = np.array([1], dtype=np.int8)

    label, _, ret, _ = triple_barrier(
        ts, px, event_idxs, tgt,
        min_ret=0.001,
        horizontal_barriers=(1.0, 1.0),
        vertical_barrier=3,
        side=side,
    )

    assert ret[0] > 0
    assert label[0] == 1                           # meta-label (good trade)


def test_no_vertical_barrier_inf():
    """
    vertical_barrier = np.inf ⇒ scan to the end of the series.
    """
    ts = _make_timestamps(6)
    px = np.array([100, 100.5, 101, 101.5, 102, 102.5], dtype=np.float64)
    event_idxs = np.array([0], dtype=np.int64)
    tgt = np.array([0.025], dtype=np.float64)

    label, t_idx, ret, _ = triple_barrier(
        ts, px, event_idxs, tgt,
        horizontal_barriers=(1.0, 1.0),
        vertical_barrier=np.inf,
        side=None,
        min_ret=0.0,
    )

    assert t_idx[0] == len(px) - 1                 # touched at the last bar
    assert label[0] == 1
    assert ret[0] > 0


# ---------------------------------------------------------------------------
# max_rb_ratios specific tests
# ---------------------------------------------------------------------------
def test_max_rb_ratio_at_barrier_hit():
    """
    Tests that max_rb_ratio is exactly 1.0 when a barrier is hit.
    """
    ts = _make_timestamps(5)
    px = np.array([100, 100.5, 101, 101.5, 102], dtype=np.float64)
    event_idxs = np.array([0], dtype=np.int64)
    tgt = np.array([0.005], dtype=np.float64)      # 0.5% target

    # Price movement will hit the upper barrier at index 1
    label, t_idx, ret, rbr = triple_barrier(
        ts, px, event_idxs, tgt,
        horizontal_barriers=(1.0, 1.0),
        vertical_barrier=4,
        side=None,
        min_ret=0.0,
    )

    # When the price hits a barrier, rbr should be exactly 1.0
    assert rbr[0] == 1.0
    # Verify the barrier was actually hit
    assert t_idx[0] < 4  # Touch before vertical barrier


def test_max_rb_ratio_below_one():
    """
    Tests that max_rb_ratio is less than 1.0 when path doesn't hit a barrier.
    """
    ts = _make_timestamps(5)
    # Price moves up but not enough to hit upper barrier
    px = np.array([100, 100.2, 100.3, 100.4, 100.3], dtype=np.float64)
    event_idxs = np.array([0], dtype=np.int64)
    tgt = np.array([0.01], dtype=np.float64)       # 1% target (won't be reached)

    label, t_idx, ret, rbr = triple_barrier(
        ts, px, event_idxs, tgt,
        horizontal_barriers=(1.0, 1.0),
        vertical_barrier=4,
        side=None,
        min_ret=0.0,
    )

    # Price doesn't hit barrier, so rbr should be less than 1.0
    assert 0.0 < rbr[0] < 1.0
    # Verify we hit the vertical barrier (since horizontal wasn't hit)
    assert t_idx[0] == 4


def test_max_rb_ratio_with_disabled_barriers():
    """
    Tests that max_rb_ratio is NaN when barriers are disabled.
    """
    ts = _make_timestamps(5)
    px = np.array([100, 101, 102, 103, 104], dtype=np.float64)
    event_idxs = np.array([0], dtype=np.int64)
    tgt = np.array([0.01], dtype=np.float64)

    # Disable horizontal barriers
    label, t_idx, ret, rbr = triple_barrier(
        ts, px, event_idxs, tgt,
        horizontal_barriers=(np.inf, np.inf),
        vertical_barrier=4,
        side=None,
        min_ret=0.0,
    )

    # When barriers are disabled, rbr should be NaN
    assert np.isnan(rbr[0])


def test_rb_ratio_calculation():
    """
    Tests that the rb_ratio is calculated correctly based on how close price gets to barrier.
    """
    ts = _make_timestamps(5)
    # Price approaches halfway to upper barrier
    px = np.array([100, 100.5, 100.4, 100.3, 100.2], dtype=np.float64)
    event_idxs = np.array([0], dtype=np.int64)
    tgt = np.array([0.01], dtype=np.float64)      # 1% target

    label, t_idx, ret, rbr = triple_barrier(
        ts, px, event_idxs, tgt,
        horizontal_barriers=(1.0, 1.0),
        vertical_barrier=4,
        side=None,
        min_ret=0.0,
    )

    # Price reached about halfway to barrier (0.5% vs 1% target)
    # The exact value depends on the implementation details
    expected_return = np.log(100.5/100)
    expected_ratio = expected_return / (tgt[0] * 1.0)  # ret/barrier

    assert rbr[0] == pytest.approx(expected_ratio, abs=0.1)
    assert 0.4 < rbr[0] < 0.6  # Should be around 0.5


def test_rb_ratio_with_sell_side():
    """
    Tests that max_rb_ratio is calculated correctly for sell side.
    """
    ts = _make_timestamps(5)
    px = np.array([100, 99.5, 99.6, 99.7, 99.8], dtype=np.float64)
    event_idxs = np.array([0], dtype=np.int64)
    tgt = np.array([0.01], dtype=np.float64)      # 1% target
    side = np.array([-1], dtype=np.int8)  # Sell side

    label, t_idx, ret, rbr = triple_barrier(
        ts, px, event_idxs, tgt,
        horizontal_barriers=(1.0, 1.0),
        vertical_barrier=4,
        side=side,
        min_ret=0.001,
    )

    # For sell side, price moved favorable direction (down) but didn't hit barrier
    assert ret[0] > 0  # Positive return on a sell when price drops
    assert 0.0 < rbr[0] < 1.0  # Should be proportional to how close it got to target
    # The exact value will depend on implementation details


# ---------------------------------------------------------------------------
# failure-mode tests
# ---------------------------------------------------------------------------
def test_invalid_lengths():
    ts = _make_timestamps(3)
    px = np.array([100, 101, 102], dtype=np.float64)
    tgt = np.array([0.005, 0.005], dtype=np.float64)   # len mismatch
    event_idxs = np.array([0], dtype=np.int64)

    with pytest.raises(ValueError, match="lengths.*must match"):
        triple_barrier(
            ts, px, event_idxs, tgt,
            horizontal_barriers=(1.0, 1.0),
            vertical_barrier=2,
            side=None,
            min_ret=0.0,
        )


@pytest.mark.parametrize("vbar", [0, -1])
def test_non_positive_vertical_barrier(vbar):
    ts = _make_timestamps(3)
    px = np.array([100, 101, 102], dtype=np.float64)
    event_idxs = np.array([0], dtype=np.int64)
    tgt = np.array([0.005], dtype=np.float64)

    with pytest.raises(ValueError, match="vertical barrier"):
        triple_barrier(
            ts, px, event_idxs, tgt,
            min_ret=0.001,
            horizontal_barriers=(1.0, 1.0),
            vertical_barrier=vbar,
            side=None,
        )


def test_negative_min_ret():
    ts = _make_timestamps(3)
    px = np.array([100, 101, 102], dtype=np.float64)
    event_idxs = np.array([0], dtype=np.int64)
    tgt = np.array([0.005], dtype=np.float64)

    with pytest.raises(ValueError, match="minimum return"):
        triple_barrier(
            ts, px, event_idxs, tgt,
            horizontal_barriers=(1.0, 1.0),
            vertical_barrier=2,
            side=None,
            min_ret=-1.0,
        )


# ---------------------------------------------------------------------------
# multiple events tests
# ---------------------------------------------------------------------------
def test_multiple_events():
    """Test processing multiple events simultaneously."""
    ts = _make_timestamps(10)
    px = np.array([100, 101, 102, 103, 102, 101, 100, 99, 98, 99], dtype=np.float64)
    event_idxs = np.array([0, 5], dtype=np.int64)  # Two events at different timestamps
    tgt = np.array([0.02, 0.02], dtype=np.float64)  # Same target for both events

    labels, t_idxs, rets, rbrs = triple_barrier(
        ts, px, event_idxs, tgt,
        horizontal_barriers=(1.0, 1.0),
        vertical_barrier=4,
        side=None,
        min_ret=0.0,
    )

    # First event should hit upper barrier
    assert labels[0] == 1
    assert t_idxs[0] < 4  # Should hit barrier before vertical
    assert rbrs[0] == 1.0  # Should have hit barrier

    # Second event should hit lower barrier
    assert labels[1] == -1
    assert t_idxs[1] < 9  # Should hit barrier before vertical
    assert rbrs[1] == 1.0  # Should have hit barrier


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
