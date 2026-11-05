import pandas as pd

TF_HOURS = {"H4": 4, "D1": 24}


def resample_ny_anchor(h1: pd.DataFrame, tf: str, anchor_hour: int = 17,
                       tz: str = "America/New_York") -> pd.DataFrame:
    """
    Build H4 / D1 candles from H1 candles, anchored to `anchor_hour` New York time
    (17 = the forex session open, which is what TradingView uses).
    Removes the Sunday stub candles a UTC-anchored broker feed produces.
    Input/output `time` is naive UTC (candle open).
    """
    if tf not in TF_HOURS:
        raise ValueError(f"resample only supports {list(TF_HOURS)}")
    freq = f"{TF_HOURS[tf]}h" if tf == "H4" else "1D"

    ny = pd.DatetimeIndex(h1["time"]).tz_localize("UTC").tz_convert(tz)
    shifted = (ny - pd.Timedelta(hours=anchor_hour)).tz_localize(None)
    bins = shifted.floor(freq)

    g = h1.assign(_bin=bins.to_numpy()).groupby("_bin", sort=True)
    out = pd.DataFrame({
        "time": g["time"].first(),          # actual UTC open of the first H1 in the bin
        "open": g["open"].first(),
        "high": g["high"].max(),
        "low": g["low"].min(),
        "close": g["close"].last(),
        "volume": g["volume"].sum(),
    }).reset_index(drop=True)
    return out