from numba import njit, prange
import numpy as np
from numpy.typing import NDArray

@njit(nogil=True, parallel=True)
def time_cues(timestamps: NDArray[np.int64]):
    """
    :param timestamps : int64 nanoseconds UTC
    :returns: tuple of arrays for the block-7 features
              (sin_td, cos_td, dow,, asia, eu, us, sess_x, top_hr)
    """
    n = timestamps.size
    # pre-allocate
    sin_td  = np.empty(n, np.float64)
    cos_td  = np.empty(n, np.float64)
    dow     = np.empty(n, np.int8)    # day of week
    asia    = np.zeros(n, np.bool_)
    eu      = np.zeros(n, np.bool_)
    us      = np.zeros(n, np.bool_)
    trans   = np.zeros(n, np.bool_)   # optional
    top_hr  = np.zeros(n, np.bool_)   # optional

    secs_day = 86400.0
    twopi    = 2.0*np.pi

    for i in prange(n):
        ts = timestamps[i] // 1_000_000_000   # seconds

        # ---------- time-of-day cyclical ----------
        sec_in_day = ts % 86400
        phase = twopi * (sec_in_day / secs_day)
        sin_td[i] = np.sin(phase)
        cos_td[i] = np.cos(phase)

        # ---------- day-of-week cyclical ----------
        # Unix epoch (Jan 1, 1970) was a Thursday
        # To get day of week where Monday=0, we offset by 3
        day_week = ((ts // 86400) + 3) % 7
        dow[i] = day_week
        # ---------- session flags ---------------
        hour = (sec_in_day // 3600)
        if 0 <= hour < 8:
            asia[i] = 1
        if 7 <= hour < 15:
            eu[i]   = 1
        if 13 <= hour < 21:
            us[i]   = 1

        # ---------- transition (optional) -------
        minute = (sec_in_day % 3600) // 60
        if (hour in (0,7,13)) and minute == 0:
            trans[i] = 1

        # ---------- on-the-hour -----------------
        if minute == 0:
            top_hr[i] = 1

    return sin_td, cos_td, dow, asia, eu, us, trans, top_hr

