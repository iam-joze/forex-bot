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