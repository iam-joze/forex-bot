import pandas as pd

SWING_COLS = ["idx", "time", "price", "kind", "confirmed_idx"]


def find_swings(df: pd.DataFrame, n: int = 2) -> pd.DataFrame:
    """
    Fractal swings. Swing high at i: high[i] is above the n candles before it and
    at least as high as the n candles after it (mirror for lows).
    A swing only becomes known when candle i+n closes -> `confirmed_idx`.
    """
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    times = df["time"].to_numpy()
    rows = []
    for i in range(n, len(df) - n):
        if high[i] > high[i - n:i].max() and high[i] >= high[i + 1:i + n + 1].max():
            rows.append((i, times[i], high[i], "high", i + n))
        if low[i] < low[i - n:i].min() and low[i] <= low[i + 1:i + n + 1].min():
            rows.append((i, times[i], low[i], "low", i + n))
    out = pd.DataFrame(rows, columns=SWING_COLS)
    return out.sort_values(["confirmed_idx", "idx"], kind="stable").reset_index(drop=True)

def alternate(swings: pd.DataFrame) -> pd.DataFrame:
    """
    Force strict high/low alternation: of consecutive same-kind swings keep only the
    most extreme one (highest high / lowest low). Confirmation is the later of the two,
    so nothing becomes known earlier than it really was.
    """
    kept = []
    for s in swings.sort_values(["idx"], kind="stable").itertuples(index=False):
        if kept and kept[-1].kind == s.kind:
            prev = kept[-1]
            better = (s.price > prev.price) if s.kind == "high" else (s.price < prev.price)
            conf = max(prev.confirmed_idx, s.confirmed_idx)
            keep = s if better else prev
            kept[-1] = keep._replace(confirmed_idx=conf)
        else:
            kept.append(s)
    out = pd.DataFrame(kept, columns=SWING_COLS)
    return out.sort_values(["confirmed_idx", "idx"], kind="stable").reset_index(drop=True)


def find_swings_zigzag(df: pd.DataFrame, atr_mult: float = 2.0, atr_period: int = 14) -> pd.DataFrame:
    """
    ZigZag swings: a swing high is confirmed once price falls `atr_mult` x ATR below
    the running high (mirror for lows). Alternates by construction; confirmed_idx is
    the candle where the reversal was big enough, so there is no lookahead.
    """
    from structure.utils import atr
    a = atr(df, atr_period).to_numpy()
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    times = df["time"].to_numpy()
    rows = []
    mode = 0                      # 0 undecided, 1 tracking a high, -1 tracking a low
    hi_i = lo_i = 0
    for i in range(len(df)):
        if a[i] != a[i]:
            continue
        if mode == 0:
            if high[i] > high[hi_i]:
                hi_i = i
            if low[i] < low[lo_i]:
                lo_i = i
            if high[hi_i] - low[i] >= atr_mult * a[i] and hi_i < i:
                rows.append((hi_i, times[hi_i], high[hi_i], "high", i))
                mode, lo_i = -1, i
            elif high[i] - low[lo_i] >= atr_mult * a[i] and lo_i < i:
                rows.append((lo_i, times[lo_i], low[lo_i], "low", i))
                mode, hi_i = 1, i
        elif mode == 1:
            if high[i] > high[hi_i]:
                hi_i = i
            elif high[hi_i] - low[i] >= atr_mult * a[i]:
                rows.append((hi_i, times[hi_i], high[hi_i], "high", i))
                mode, lo_i = -1, i
        else:
            if low[i] < low[lo_i]:
                lo_i = i
            elif high[i] - low[lo_i] >= atr_mult * a[i]:
                rows.append((lo_i, times[lo_i], low[lo_i], "low", i))
                mode, hi_i = 1, i
    return pd.DataFrame(rows, columns=SWING_COLS)