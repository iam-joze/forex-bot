import numpy as np
import pandas as pd
from structure.utils import atr

EVENT_COLS = ["idx", "time", "direction", "type", "broken_idx", "level",
              "origin_idx", "leg_atr"]


def find_structure(df: pd.DataFrame, swings: pd.DataFrame, atr_period: int = 14) -> pd.DataFrame:
    """
    Walks candle by candle (no lookahead) and emits BOS / MSS events.

    - A break = a candle CLOSE beyond the most recent confirmed swing level.
    - BOS: break in the direction of the current trend.  MSS: break against it.
    - Trend becomes the break direction. (First break after start has no prior trend -> BOS.)
    - origin_idx: start of the impulse leg that made the break (extreme between the
      broken swing and the break candle).
    - leg_atr: size of that leg in ATRs, so "impulsive" can be thresholded later.
    """
    close = df["close"].to_numpy()
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    times = df["time"].to_numpy()
    a = atr(df, atr_period).to_numpy()

    by_conf = {}
    for s in swings.itertuples(index=False):
        by_conf.setdefault(int(s.confirmed_idx), []).append(s)

    last_high = None  # (idx, price) of most recent unbroken confirmed swing high
    last_low = None
    trend = None
    rows = []

    for i in range(len(df)):
        for s in by_conf.get(i, []):
            if s.kind == "high":
                last_high = (int(s.idx), float(s.price))
            else:
                last_low = (int(s.idx), float(s.price))

        if last_high is not None and close[i] > last_high[1]:
            h_idx, level = last_high
            origin = h_idx + int(np.argmin(low[h_idx:i + 1]))
            etype = "MSS" if trend == "bear" else "BOS"
            leg = (close[i] - low[origin]) / a[i] if a[i] == a[i] and a[i] > 0 else np.nan
            rows.append((i, times[i], "bull", etype, h_idx, level, origin, leg))
            trend, last_high = "bull", None

        elif last_low is not None and close[i] < last_low[1]:
            l_idx, level = last_low
            origin = l_idx + int(np.argmax(high[l_idx:i + 1]))
            etype = "MSS" if trend == "bull" else "BOS"
            leg = (high[origin] - close[i]) / a[i] if a[i] == a[i] and a[i] > 0 else np.nan
            rows.append((i, times[i], "bear", etype, l_idx, level, origin, leg))
            trend, last_low = "bear", None

    return pd.DataFrame(rows, columns=EVENT_COLS)