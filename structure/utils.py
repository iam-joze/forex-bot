import pandas as pd


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Simple-average ATR. Value at row i uses candles up to and including i."""
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()