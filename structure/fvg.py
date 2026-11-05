import numpy as np
import pandas as pd
from structure.utils import atr

RAW_COLS = ["c1", "c2", "c3", "time", "direction", "bottom", "top", "size"]


def find_fvgs(df: pd.DataFrame, min_atr_mult: float = 0.3, atr_period: int = 14) -> pd.DataFrame:
    """
    Raw 3-candle imbalances (no structure rules yet).
      bull: low[c3] > high[c1]   zone = [high[c1], low[c3]]
      bear: high[c3] < low[c1]   zone = [high[c3], low[c1]]
    Size filter is relative: gap >= min_atr_mult * ATR (at c3).
    """
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    times = df["time"].to_numpy()
    a = atr(df, atr_period).to_numpy()
    rows = []
    for c2 in range(1, len(df) - 1):
        c1, c3 = c2 - 1, c2 + 1
        if low[c3] > high[c1]:
            direction, bottom, top = "bull", high[c1], low[c3]
        elif high[c3] < low[c1]:
            direction, bottom, top = "bear", high[c3], low[c1]
        else:
            continue
        size = top - bottom
        if not (a[c3] == a[c3]) or size < min_atr_mult * a[c3]:
            continue
        rows.append((c1, c2, c3, times[c2], direction, bottom, top, size))
    return pd.DataFrame(rows, columns=RAW_COLS)


def validate_fvgs(fvgs: pd.DataFrame, events: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """
    Keeps only FVGs that START the impulse leg of a BOS/MSS (same direction):
        c1 >= leg origin  and  displacement candle c2 <= break candle.
    FVG is 'known' from max(break candle, c3).
    Touches before that are ignored (still valid). The first touch AFTER that
    is the mitigation. Untouched = 'active'.
    """
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    times = df["time"].to_numpy()
    n = len(df)
    ev = list(events.itertuples(index=False))
    per_event = {}
    rows = []

    for f in fvgs.itertuples(index=False):
        match = None
        for e in ev:
            if e.direction == f.direction and f.c1 >= e.origin_idx and f.c2 <= e.idx:
                match = e
                break
        if match is None:
            continue

        known = max(int(match.idx), int(f.c3))
        mit = None
        for j in range(known + 1, n):
            if f.direction == "bull" and low[j] <= f.top:
                mit = j
                break
            if f.direction == "bear" and high[j] >= f.bottom:
                mit = j
                break

        order = per_event.get(int(match.idx), 0)
        per_event[int(match.idx)] = order + 1
        rows.append({
            "time": f.time, "direction": f.direction,
            "bottom": f.bottom, "top": f.top, "size": f.size,
            "c1": f.c1, "c2": f.c2, "c3": f.c3,
            "event_type": match.type, "break_idx": int(match.idx),
            "break_time": match.time, "leg_order": order,
            "known_idx": known, "known_time": times[known],
            "mitigated_idx": mit,
            "mitigated_time": times[mit] if mit is not None else pd.NaT,
            "status": "mitigated" if mit is not None else "active",
        })
    return pd.DataFrame(rows)