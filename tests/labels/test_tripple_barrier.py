import numpy as np
import pytest
import os
# disable jit
os.environ['NUMBA_DISABLE_JIT'] = "1"

from finmlkit.label.tbm import triple_barrier

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
NS = 1_000_000_000          # seconds → nanoseconds


def _make_timestamps(n):
    """Regular one-second grid in nanoseconds."""
    return (np.arange(n, dtype=np.int64) * NS)


# ---------------------------------------------------------------------------
# success-path tests
# ---------------------------------------------------------------------------
def test_upper_barrier_hit():
    """
    Price rises immediately – upper barrier should be hit on the first bar.
    """
    ts   = _make_timestamps(5)
    px   = np.array([100, 101, 101, 101, 101], dtype=np.float64)
    event_ts = np.array([ts[0]], dtype=np.int64)  # Using timestamp instead of index
    tgt  = np.array([0.005], dtype=np.float64)    # Target for the single event

    label, t_idx, ret, rbr = triple_barrier(
        ts, px, event_ts, tgt,
        min_ret=0.001,
        horizontal_barriers=(1.0, 1.0),
        vertical_barrier=3,                        # sec
        side=None,
    )

    expected_ret = np.log(101 / 100.0)
    assert label[0] == 1
    assert t_idx[0] == 1
    assert ret[0] == pytest.approx(expected_ret)
    assert rbr[0] >= 1.0                           # should have crossed


def test_lower_barrier_hit():
    """
    Price drops – lower barrier hit.
    """
    ts   = _make_timestamps(5)
    px   = np.array([100, 99, 99, 99, 99], dtype=np.float64)
    event_ts = np.array([ts[0]], dtype=np.int64)  # Using timestamp instead of index
    tgt  = np.array([0.005], dtype=np.float64)     # Target for the single event

    label, t_idx, ret, rbr = triple_barrier(
        ts, px, event_ts, tgt,
        min_ret=0.001,
        horizontal_barriers=(1.0, 1.0),
        vertical_barrier=3,
        side=None,
    )

    expected_ret = np.log(99 / 100.0)
    assert label[0] == -1
    assert t_idx[0] == 1
    assert ret[0] == pytest.approx(expected_ret)
    assert rbr[0] >= 1.0


def test_vertical_barrier_only():
    """
    No horizontal barriers ⇒ label comes from the vertical timeout.
    """
    ts   = _make_timestamps(5)
    px   = np.array([100, 100.2, 100.3, 100.25, 100.2], dtype=np.float64)
    event_ts = np.array([ts[0]], dtype=np.int64)  # Using timestamp instead of index
    tgt  = np.array([0.002], dtype=np.float64)      # Target for the single event (0.2%)

    label, t_idx, ret, rbr = triple_barrier(
        ts, px, event_ts, tgt,
        min_ret=0.001,
        horizontal_barriers=(np.inf, np.inf),      # disabled
        vertical_barrier=3,
        side=None,
    )

    assert t_idx[0] == 3                           # 3 s timeout
    assert rbr[0] != rbr[0]                       # nan when barriers disabled
    assert label[0] == 1                          # positive return
    assert ret[0] > 0


def test_min_ret_filter():
    """
    Targets below min_ret must be skipped (label 0, idx -1, NaNs).
    """
    ts   = _make_timestamps(3)
    px   = np.array([100, 101, 102], dtype=np.float64)
    event_ts = np.array([ts[0]], dtype=np.int64)  # Using timestamp instead of index
    tgt  = np.array([0.0005], dtype=np.float64)   # 0.05% (< min_ret)

    label, t_idx, ret, rbr = triple_barrier(
        ts, px, event_ts, tgt,
        min_ret=0.001,
        horizontal_barriers=(1.0, 1.0),
        vertical_barrier=2,
        side=None,
    )

    assert label[0] == 0
    assert t_idx[0] == -1
    assert np.isnan(ret[0]) and np.isnan(rbr[0])


def test_meta_labeling():
    """
    With side information the output label must be (0, 1).
    """
    ts   = _make_timestamps(4)
    px   = np.array([100, 101, 102, 103], dtype=np.float64)
    event_ts = np.array([ts[0]], dtype=np.int64)  # Using timestamp instead of index
    tgt  = np.array([0.004], dtype=np.float64)
    side = np.array([1], dtype=np.int8)

    label, _, ret, _ = triple_barrier(
        ts, px, event_ts, tgt,
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
    ts   = _make_timestamps(6)
    px   = np.array([100, 100.5, 101, 101.5, 102, 102.5], dtype=np.float64)
    event_ts = np.array([ts[0]], dtype=np.int64)  # Using timestamp instead of index
    tgt  = np.array([0.025], dtype=np.float64)

    label, t_idx, ret, _ = triple_barrier(
        ts, px, event_ts, tgt,
        min_ret=0.001,
        horizontal_barriers=(1.0, 1.0),
        vertical_barrier=np.inf,
        side=None,
    )

    assert t_idx[0] == len(px) - 1                 # touched at the last bar
    assert label[0] == 1
    assert ret[0] > 0


# ---------------------------------------------------------------------------
# failure-mode tests
# ---------------------------------------------------------------------------
def test_invalid_lengths():
    ts   = _make_timestamps(3)
    px   = np.array([100, 101, 102], dtype=np.float64)
    tgt  = np.array([0.005, 0.005], dtype=np.float64)   # len mismatch
    event_ts = np.array([ts[0]], dtype=np.int64)  # Using timestamp instead of index

    with pytest.raises(ValueError, match="lengths.*must match"):
        triple_barrier(
            ts, px, event_ts, tgt,
            min_ret=0.001,
            horizontal_barriers=(1.0, 1.0),
            vertical_barrier=2,
            side=None,
        )


@pytest.mark.parametrize("vbar", [0, -1])
def test_non_positive_vertical_barrier(vbar):
    ts   = _make_timestamps(3)
    px   = np.array([100, 101, 102], dtype=np.float64)
    event_ts = np.array([ts[0]], dtype=np.int64)  # Using timestamp instead of index
    tgt  = np.array([0.005], dtype=np.float64)

    with pytest.raises(ValueError, match="vertical barrier"):
        triple_barrier(
            ts, px, event_ts, tgt,
            min_ret=0.001,
            horizontal_barriers=(1.0, 1.0),
            vertical_barrier=vbar,
            side=None,
        )


def test_negative_min_ret():
    ts   = _make_timestamps(3)
    px   = np.array([100, 101, 102], dtype=np.float64)
    event_ts = np.array([ts[0]], dtype=np.int64)  # Using timestamp instead of index
    tgt  = np.array([0.005], dtype=np.float64)

    with pytest.raises(ValueError, match="minimum return"):
        triple_barrier(
            ts, px, event_ts, tgt,
            min_ret=-0.1,
            horizontal_barriers=(1.0, 1.0),
            vertical_barrier=2,
            side=None,
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
    #pytest.main([__file__, "-s", "--tb=short"])  # for debugging, to see print statements
