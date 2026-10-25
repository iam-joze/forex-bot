from datetime import datetime
import pandas as pd
from broker.base import Broker
from config.settings import PairConfig, ROLES
from data.timeframes import tf_delta


def load_mtf(broker: Broker, pair: PairConfig, start: datetime, end: datetime,
             warmup_bars: int = 100) -> dict:
    """Fetch every role's timeframe, with extra history so higher TFs have context at `start`."""
    frames = {}
    for role in ROLES:
        tf = pair.tf(role)
        warm_start = start - tf_delta(tf) * warmup_bars
        frames[role] = broker.get_candles_range(pair.broker_symbol, tf, warm_start, end)
    return frames


def align_mtf(frames: dict, pair: PairConfig, start: datetime,
              base_role: str = "entry") -> pd.DataFrame:
    """
    One row per entry-role candle. For each other role r, adds:
      {r}_parent_time    open time of the HTF candle this candle sits inside
      {r}_closed_*       OHLC of the latest HTF candle that was already CLOSED
                         when this candle closed (no lookahead)
      {r}_closed_at      that HTF candle's close time
    """
    out = frames[base_role].copy()
    out["close_time"] = out["time"] + tf_delta(pair.tf(base_role))

    for role in ROLES:
        if role == base_role:
            continue
        htf = frames[role].copy()
        htf["close_time"] = htf["time"] + tf_delta(pair.tf(role))

        parents = htf[["time"]].rename(columns={"time": f"{role}_parent_time"})
        out = pd.merge_asof(out, parents, left_on="time",
                            right_on=f"{role}_parent_time", direction="backward")

        closed = htf[["close_time", "open", "high", "low", "close"]].rename(columns={
            "close_time": f"{role}_closed_at",
            "open": f"{role}_closed_open", "high": f"{role}_closed_high",
            "low": f"{role}_closed_low", "close": f"{role}_closed_close",
        })
        out = pd.merge_asof(out, closed, left_on="close_time",
                            right_on=f"{role}_closed_at", direction="backward")

    return out[out["time"] >= pd.Timestamp(start)].dropna().reset_index(drop=True)