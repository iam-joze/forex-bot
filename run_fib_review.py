"""
Fib-zone review over LONG history. For every order block / validated FVG it decides:
  - does its direction match the fib of the latest BOS/MSS leg?
  - is it on the right side of 0.5 (bullish in discount, bearish in premium)?
  - does it touch the 0.62-0.79 band (noted only)?
Prints stats and saves sample charts (right-side and wrong-side examples) to logs/fib/.
"""
import os
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from structure.swings import find_swings, alternate
from structure.choch_mss import find_structure
from structure.fvg import find_fvgs, validate_fvgs
from structure.order_blocks import find_order_blocks
from structure.fib_zones import fib_state, evaluate_pois
from tools.plot import plot_structure

PAIR = "EURUSD"
HISTORY_DAYS = {"D1": 1460, "H4": 1095, "H1": 730, "M15": 365}
SWING_N = 5
FVG_MIN_ATR = 0.3
MIN_LEG_ATR = 2.0
FIB_CARRY_OVER = False   # True = keep the previous fib until the newest leg's swings are marked
RIGHT_SAMPLES = 4    # charts of POIs on the right side
WRONG_SAMPLES = 2    # charts of POIs rejected for being on the wrong side of 0.5
MIN_CANDLES, MAX_CANDLES, CONTEXT = 160, 220, 40


def _spread(frame, k):
    if frame.empty or k <= 0:
        return frame.iloc[0:0]
    return frame.iloc[np.unique(np.linspace(0, len(frame) - 1, min(k, len(frame))).astype(int))]


def review(df, tf, pair, outdir="logs/fib"):
    df = df.reset_index(drop=True)
    sw = alternate(find_swings(df, SWING_N))
    ev = find_structure(df, sw)
    fv = validate_fvgs(find_fvgs(df, FVG_MIN_ATR), ev, df)
    ob = find_order_blocks(df, ev, min_leg_atr=MIN_LEG_ATR)
    fib = fib_state(df, ev, sw, carry_over=FIB_CARRY_OVER)
    pois = evaluate_pois(df, fib, ob, fv)

    print(f"\n=== {pair} {tf}: {len(df)} candles, {df['time'].iloc[0]:%Y-%m-%d} -> {df['time'].iloc[-1]:%Y-%m-%d} ===")
    st = fib.attrs["stats"]
    print(f"BOS/MSS legs {st['legs']} -> legs with a fib on marked swings {st['legs_with_fib']};  "
          f"candles with a fib in force: {fib['dir'].notna().mean():.0%}")
    if pois.empty:
        print("no POIs")
        return pois
    for kind in ("OB", "FVG"):
        p = pois[pois.kind == kind]
        if p.empty:
            continue
        j = p[p.side != "no_fib"]
        m = j[j.trend_match]
        rt = j[j.right_side]
        print(f"{kind}: {len(p)} total | no fib yet at first touch {len(p) - len(j)} | judged {len(j)} | "
              f"trend matches {len(m)} | right side of 0.5 {len(rt)}; straddling 0.5: "
              f"{int((m.side == 'straddles').sum())}, wrong side: "
              f"{int(((m.side != 'straddles') & ~m.right_side).sum())}")
        if len(rt):
            print(f"     of the right-side {kind}s, 0.62-0.79 band: inside {(rt.ote == 'inside').mean():.0%}, "
                  f"overlaps {(rt.ote == 'overlaps').mean():.0%}, none {(rt.ote == 'none').mean():.0%}")

    right = pois[pois.right_side].sort_values("eval_idx")
    wrong = pois[pois.trend_match & ~pois.right_side & ~pois.side.isin(["straddles", "no_fib"])].sort_values("eval_idx")
    os.makedirs(outdir, exist_ok=True)
    jobs = [("ok", r) for r in _spread(right, RIGHT_SAMPLES).itertuples(index=False)] + \
           [("wrong", r) for r in _spread(wrong, WRONG_SAMPLES).itertuples(index=False)]
    for k, (tag, p) in enumerate(jobs, 1):
        e = min(len(df), max(int(p.eval_idx), int(p.fib_draw_to)) + 20)
        s = max(0, min(int(p.idx), int(p.fib_start_idx)) - CONTEXT)
        if e - s > MAX_CANDLES:
            s = e - MAX_CANDLES
        if e - s < MIN_CANDLES:
            s = max(0, e - MIN_CANDLES)
            e = min(len(df), s + MIN_CANDLES)
        fibs = [dict(x_start=int(p.fib_start_idx), x_end=int(p.fib_end_idx), x_to=int(p.fib_draw_to),
                     start=p.fib_start, end=p.fib_end, mid=p.mid, lvl62=p.lvl62, lvl79=p.lvl79,
                     ote_lo=p.ote_lo, ote_hi=p.ote_hi)]
        path = f"{outdir}/{pair}_{tf}_{k}_{tag}_{p.kind}_{pd.Timestamp(df['time'].iloc[int(p.idx)]):%Y%m%d_%H%M}.png"
        plot_structure(df, sw, ev, fv, s, e, path,
                       f"{pair} {tf} {p.direction} {p.kind} | fib {p.fib_dir} | {p.side} of 0.5 | "
                       f"0.62-0.79: {p.ote} | {'RIGHT side' if tag == 'ok' else 'WRONG side'}",
                       obs=ob, fibs=fibs)
        print("   chart:", path)
    return pois


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