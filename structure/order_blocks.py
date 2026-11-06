import numpy as np
import pandas as pd
from structure.swings import find_swings

OB_COLS = ["idx", "time", "direction", "high", "low", "body_high", "body_low",
           "event_type", "break_idx", "break_time", "leg_atr",
           "fvg_bottom", "fvg_top", "known_idx", "known_time",
           "mitigated_idx", "mitigated_time", "status", "liq_grab", "liq_level"]


def find_order_blocks(df: pd.DataFrame, events: pd.DataFrame,
                      min_leg_atr: float = 2.0, max_after_origin: int = 5,
                      bear_close_ref: str = "open",
                      liq_lookback: int = 40, liq_swing_n: int = 2) -> pd.DataFrame:
    """
    One order block per BOS/MSS leg: the FIRST candle c (scanning from just before the leg's
    origin) that is the last opposite-colour candle before the impulse:

    Bullish OB (red candle before a bullish BOS/MSS):
      - c is bearish; its low clears the previous candle's low (wicks count)
      - the next candle does not trade below c's low
      - the next candle is bullish and CLOSES ABOVE c's high
      - that leaves an FVG of any size: low[c+2] > high[c]
    Bearish OB (green candle before a bearish BOS/MSS):
      - c is bullish; its high clears the previous candle's high
      - the next candle does not trade above c's high
      - the next candle is bearish and CLOSES BELOW c's open (bear_close_ref="open";
        use "low" to require a close below c's low instead)
      - that leaves an FVG of any size: high[c+2] < low[c]
    The leg must also be a strong impulse (leg_atr >= min_leg_atr).
    c may sit up to max_after_origin candles after the leg's origin.

    Noted only, never required: liq_grab = c's wick took out a fresh swing level
    (raw swing_n=2 swing confirmed before c, within liq_lookback candles, not already swept).
    known_idx = max(break candle, c+2). Mitigation = first touch of the OB's near edge
    (full range, wicks included) after known_idx.
    df.attrs['funnel'] = number of legs surviving each step.
    """
    open_ = df["open"].to_numpy()
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    close = df["close"].to_numpy()
    times = df["time"].to_numpy()
    n = len(df)

    raw = find_swings(df, liq_swing_n)
    sw = {k: (g["idx"].to_numpy(), g["confirmed_idx"].to_numpy(), g["price"].to_numpy())
          for k, g in raw.groupby("kind")}

    funnel = {"legs": 0, "strong": 0, "ob": 0}
    rows = []
    for e in events.itertuples(index=False):
        funnel["legs"] += 1
        if not (e.leg_atr >= min_leg_atr):          # NaN also fails
            continue
        funnel["strong"] += 1

        bull = e.direction == "bull"
        o = int(e.origin_idx)
        found = None
        for c in range(max(1, o - 1), min(o + max_after_origin, int(e.idx) - 1) + 1):
            if c + 2 >= n:
                break
            if bull:
                ok = (close[c] < open_[c] and low[c] < low[c - 1] and low[c + 1] >= low[c]
                      and close[c + 1] > open_[c + 1] and close[c + 1] > high[c]
                      and low[c + 2] > high[c])
            else:
                ref = open_[c] if bear_close_ref == "open" else low[c]
                ok = (close[c] > open_[c] and high[c] > high[c - 1] and high[c + 1] <= high[c]
                      and close[c + 1] < open_[c + 1] and close[c + 1] < ref
                      and high[c + 2] < low[c])
            if ok:
                found = c
                break
        if found is None:
            continue
        c = found
        funnel["ob"] += 1

        known = max(int(e.idx), c + 2)
        mit = None
        for j in range(known + 1, n):
            if (bull and low[j] <= high[c]) or (not bull and high[j] >= low[c]):
                mit = j
                break

        liq, level = False, np.nan
        kind = "low" if bull else "high"
        if kind in sw:
            s_idx, s_conf, s_px = sw[kind]
            m = (s_conf < c) & (c - s_idx <= liq_lookback) & (s_idx < c)
            m &= (s_px > low[c]) if bull else (s_px < high[c])
            for k in np.where(m)[0][::-1]:                       # most recent first
                seg = low[s_idx[k] + 1:c] if bull else high[s_idx[k] + 1:c]
                fresh = (len(seg) == 0) or ((seg.min() >= s_px[k]) if bull else (seg.max() <= s_px[k]))
                if fresh:
                    liq, level = True, float(s_px[k])
                    break

        fb, ft = (high[c], low[c + 2]) if bull else (high[c + 2], low[c])
        rows.append((c, times[c], e.direction, high[c], low[c],
                     max(open_[c], close[c]), min(open_[c], close[c]),
                     e.type, int(e.idx), e.time, e.leg_atr, fb, ft,
                     known, times[known], mit, times[mit] if mit is not None else pd.NaT,
                     "mitigated" if mit is not None else "active", liq, level))

    out = pd.DataFrame(rows, columns=OB_COLS)
    out.attrs["funnel"] = funnel
    return out