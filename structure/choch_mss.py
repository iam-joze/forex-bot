import numpy as np
import pandas as pd
from structure.utils import atr

EVENT_COLS = ["idx", "time", "direction", "type", "broken_idx", "level",
              "origin_idx", "leg_atr"]


def find_structure(df: pd.DataFrame, swings: pd.DataFrame, atr_period: int = 14) -> pd.DataFrame:
    """
    Walks candle by candle (no lookahead) and emits BOS / MSS events.

    - BOS (break in the direction of the current trend): a WICK beyond the most recent confirmed
      swing level is enough (high above a swing high in an uptrend, low below a swing low in a downtrend).
    - MSS (break against the trend): needs a candle CLOSE beyond the swing level.
    - Trend becomes the break direction. (First break after start has no prior trend -> BOS, wick is enough.)
    - If one candle qualifies on both sides: the side its close is beyond wins; if neither, the candle colour decides.
    - origin_idx: start of the impulse leg that made the break (extreme between the
      broken swing and the break candle).
    - leg_atr: size of that leg in ATRs (measured from the close, as before).
    """
    open_ = df["open"].to_numpy()
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

        bull_ok = last_high is not None and (close[i] > last_high[1] if trend == "bear"
                                             else high[i] > last_high[1])
        bear_ok = last_low is not None and (close[i] < last_low[1] if trend == "bull"
                                            else low[i] < last_low[1])
        if bull_ok and bear_ok:
            if close[i] > last_high[1]:
                bear_ok = False
            elif close[i] < last_low[1]:
                bull_ok = False
            elif close[i] >= open_[i]:
                bear_ok = False
            else:
                bull_ok = False

        if bull_ok:
            h_idx, level = last_high
            origin = h_idx + int(np.argmin(low[h_idx:i + 1]))
            etype = "MSS" if trend == "bear" else "BOS"
            leg = (close[i] - low[origin]) / a[i] if a[i] == a[i] and a[i] > 0 else np.nan
            rows.append((i, times[i], "bull", etype, h_idx, level, origin, leg))
            trend, last_high = "bull", None

        elif bear_ok:
            l_idx, level = last_low
            origin = l_idx + int(np.argmax(high[l_idx:i + 1]))
            etype = "MSS" if trend == "bull" else "BOS"
            leg = (high[origin] - close[i]) / a[i] if a[i] == a[i] and a[i] > 0 else np.nan
            rows.append((i, times[i], "bear", etype, l_idx, level, origin, leg))
            trend, last_low = "bear", None

    return pd.DataFrame(rows, columns=EVENT_COLS)