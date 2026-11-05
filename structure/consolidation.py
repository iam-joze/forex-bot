import numpy as np
import pandas as pd
from structure.utils import atr

RANGE_COLS = ["start_idx", "end_idx", "start_time", "end_time", "known_idx",
              "top", "bottom", "bars", "width_atr", "status",
              "break_dir", "break_idx", "break_time", "momentum", "beyond_atr"]


def find_ranges(df: pd.DataFrame,
                min_bars: int = 20,
                max_width_atr: float = 4.0,
                max_efficiency: float = 0.25,
                touch_frac: float = 0.2,
                min_touches: int = 2,
                expire_width_atr: float = 7.0,
                min_break_momentum: float = 0.7,
                min_break_beyond_atr: float = 0.2,
                atr_period: int = 14) -> pd.DataFrame:
    """
    Consolidation (range) detector, candle by candle (no lookahead).

    A range is DETECTED at candle i when the last `min_bars` candles:
      - fit in a box no taller than `max_width_atr` x ATR,
      - show no directional progress (efficiency = net move / path travelled <= max_efficiency),
      - touched both edges at least `min_touches` times (within `touch_frac` of box height).
    `known_idx` = i, the first candle where a strategy could know about it.

    While active, wicks that close back inside widen the box. The range ENDS when:
      - a candle CLOSES beyond a boundary with real momentum (body >= `min_break_momentum` x ATR
        and close >= `min_break_beyond_atr` x ATR past the edge) -> status 'broken'. Weaker closes
        beyond the edge are treated as pokes: the box simply widens.
        momentum = body / ATR, beyond_atr = distance of the close past the boundary in ATRs
      - the box grows past `expire_width_atr` x ATR -> status 'expired'
      - data ends -> status 'active'
    """
    open_ = df["open"].to_numpy()
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    close = df["close"].to_numpy()
    times = df["time"].to_numpy()
    a = atr(df, atr_period).to_numpy()
    n = len(df)
    rows = []

    def emit(start, end, known, top, bottom, status, bdir=None, bidx=None, mom=np.nan, beyond=np.nan):
        ref_atr = a[known]
        rows.append((start, end, times[start], times[end], known, top, bottom, end - start + 1,
                     (top - bottom) / ref_atr if ref_atr > 0 else np.nan, status,
                     bdir, bidx, times[bidx] if bidx is not None else pd.NaT, mom, beyond))

    active = False
    blocked_until = -1          # a new range may not overlap the previous one
    start = known = 0
    top = bottom = 0.0

    for i in range(n):
        if a[i] != a[i]:
            continue
        if not active:
            s = i - min_bars + 1
            if s < 0 or s <= blocked_until:
                continue
            hi, lo = high[s:i + 1].max(), low[s:i + 1].min()
            width = hi - lo
            if width <= 0 or width > max_width_atr * a[i]:
                continue
            path = np.abs(np.diff(close[s:i + 1])).sum()
            eff = abs(close[i] - close[s]) / path if path > 0 else 1.0
            if eff > max_efficiency:
                continue
            tol = touch_frac * width
            if (high[s:i + 1] >= hi - tol).sum() < min_touches or (low[s:i + 1] <= lo + tol).sum() < min_touches:
                continue
            active, start, known, top, bottom = True, s, i, hi, lo
        else:
            up, down = close[i] > top, close[i] < bottom
            mom = abs(close[i] - open_[i]) / a[i]
            beyond = ((close[i] - top) if up else (bottom - close[i])) / a[i] if (up or down) else 0.0
            if (up or down) and mom >= min_break_momentum and beyond >= min_break_beyond_atr:
                emit(start, i - 1, known, top, bottom, "broken", "up" if up else "down", i, mom, beyond)
                active, blocked_until = False, i
            else:
                top, bottom = max(top, high[i]), min(bottom, low[i])
                if (top - bottom) > expire_width_atr * a[i]:
                    emit(start, i - 1, known, top, bottom, "expired")
                    active, blocked_until = False, i - 1

    if active:
        emit(start, n - 1, known, top, bottom, "active")
    return pd.DataFrame(rows, columns=RANGE_COLS)