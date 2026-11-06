"""
Order-block review over LONG history. Prints how many BOS/MSS legs survive each rule and saves
sample charts spread across the whole history to logs/obs/.
"""
import os
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from structure.swings import find_swings, alternate
from structure.choch_mss import find_structure
from structure.fvg import find_fvgs, validate_fvgs
from structure.order_blocks import find_order_blocks
from tools.plot import plot_structure

PAIR = "EURUSD"
HISTORY_DAYS = {"D1": 1460, "H4": 1095, "H1": 730, "M15": 365}
SWING_N = 5
FVG_MIN_ATR = 0.3
MIN_LEG_ATR = 2.0
SAMPLES = 6
CONTEXT = 80
MIN_CANDLES = 160     # every chart shows at least this many candles
MAX_CANDLES = 220

def review(df, tf, pair, samples=SAMPLES, context=CONTEXT, outdir="logs/obs"):
    df = df.reset_index(drop=True)
    sw = alternate(find_swings(df, SWING_N))
    ev = find_structure(df, sw)
    fv = validate_fvgs(find_fvgs(df, FVG_MIN_ATR), ev, df)
    ob = find_order_blocks(df, ev, min_leg_atr=MIN_LEG_ATR)
    f = ob.attrs["funnel"]

    print(f"\n=== {pair} {tf}: {len(df)} candles, {df['time'].iloc[0]:%Y-%m-%d} -> {df['time'].iloc[-1]:%Y-%m-%d} ===")
    print(f"BOS/MSS legs {f['legs']} -> strong impulse (>= {MIN_LEG_ATR} ATR) {f['strong']} "
          f"-> order block found {f['ob']}")
    if ob.empty:
        print("no order blocks found")
        return ob
    print(f"order blocks: {len(ob)}  (bullish {int((ob.direction == 'bull').sum())}, "
          f"bearish {int((ob.direction == 'bear').sum())})")
    print(f"with a liquidity grab (noted only): {ob.liq_grab.mean():.0%};   "
          f"mitigated later: {(ob.status == 'mitigated').mean():.0%}")

    os.makedirs(outdir, exist_ok=True)
    picks = np.unique(np.linspace(0, len(ob) - 1, min(samples, len(ob))).astype(int))
    for k, i in enumerate(picks, 1):
        o = ob.iloc[i]
        end_ref = o.mitigated_idx if o.mitigated_idx == o.mitigated_idx else o.break_idx
        s = max(0, int(o.idx) - context)
        e = min(len(df), int(max(end_ref, o.break_idx)) + 40, s + MAX_CANDLES)
        if e - s < MIN_CANDLES:                      # never zoom in too far
            e = min(len(df), s + MIN_CANDLES)
            s = max(0, e - MIN_CANDLES)
        path = f"{outdir}/{pair}_{tf}_{k}_{pd.Timestamp(o.time):%Y%m%d_%H%M}.png"
        plot_structure(df, sw, ev, fv, s, e, path,
                       f"{pair} {tf} {o.direction} OB {pd.Timestamp(o.time):%Y-%m-%d %H:%M} "
                       f"({o.event_type}, leg {o.leg_atr:.1f} ATR, {o.status}"
                       f"{', liquidity grab' if o.liq_grab else ''})", obs=ob)
        print("   chart:", path)
    return ob


if __name__ == "__main__":
    from broker.mt5 import MT5Broker
    from config.settings import load_settings
    from data.loader import fetch_tf

    pair = load_settings().pairs[PAIR]
    broker = MT5Broker()
    broker.connect()
    try:
        end = datetime.now(timezone.utc).replace(tzinfo=None)
        for tf, days in HISTORY_DAYS.items():
            df = fetch_tf(broker, pair.broker_symbol, tf, end - timedelta(days=days), end)
            review(df, tf, PAIR)
    finally:
        broker.disconnect()