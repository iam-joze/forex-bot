"""
POI-validity review with the PROVISIONAL fib (no waiting for the 5-candle swing delay).
A POI is valid once its zone sits on the right side of its leg's current 0.5 (bullish below,
bearish above), and is entered the first time price touches it after that.
Prints stats and saves sample charts to logs/poi/.
"""
import os
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from structure.swings import find_swings, alternate
from structure.choch_mss import find_structure
from structure.fvg import find_fvgs, validate_fvgs
from structure.order_blocks import find_order_blocks
from structure.fib_zones import provisional_fib, validate_pois
from tools.plot import plot_structure

PAIR = "XAUUSD"
HISTORY_DAYS = {"D1": 1460, "H4": 1095, "H1": 730, "M15": 365}
SWING_N = 5
FVG_MIN_ATR = 0.3
MIN_LEG_ATR = 2.0
DOUBLE_TOP_TOL_ATR = 0.3
EXPIRE_ON_NEXT_EVENT = True   # False = a valid POI can still be entered after later BOS/MSS (until its leg dies)
VALID_SAMPLES = 4      # charts of valid, touched POIs
NEVER_SAMPLES = 2      # charts of POIs that never sat on the right side of 0.5
MIN_CANDLES, MAX_CANDLES, CONTEXT = 160, 220, 40


def _spread(frame, k):
    if frame.empty or k <= 0:
        return frame.iloc[0:0]
    return frame.iloc[np.unique(np.linspace(0, len(frame) - 1, min(k, len(frame))).astype(int))]


def review(df, tf, pair, outdir="logs/poi"):
    df = df.reset_index(drop=True)
    sw = alternate(find_swings(df, SWING_N))
    ev = find_structure(df, sw)
    fv = validate_fvgs(find_fvgs(df, FVG_MIN_ATR), ev, df)
    ob = find_order_blocks(df, ev, min_leg_atr=MIN_LEG_ATR)
    fib = provisional_fib(df, ev, sw, DOUBLE_TOP_TOL_ATR)
    pois = validate_pois(df, fib, ob, fv, expire_on_next_event=EXPIRE_ON_NEXT_EVENT)
    alt = validate_pois(df, fib, ob, fv, expire_on_next_event=not EXPIRE_ON_NEXT_EVENT)

    print(f"\n=== {pair} {tf}: {len(df)} candles, {df['time'].iloc[0]:%Y-%m-%d} -> {df['time'].iloc[-1]:%Y-%m-%d} ===")
    print(f"BOS/MSS legs {len(ev)}; candles with a fib in force: {fib['dir'].notna().mean():.0%}")
    if pois.empty:
        print("no POIs")
        return pois
    for kind in ("OB", "FVG"):
        p = pois[pois.kind == kind]
        if p.empty:
            continue
        v = p[p.status != "never_valid"]
        t = v[v.status == "touched"]
        print(f"{kind}: {len(p)} total | never on the right side of 0.5: {len(p) - len(v)} | valid: {len(v)} "
              f"(touched after becoming valid {len(t)}, still untouched {len(v) - len(t)})")
        if len(v):
            print(f"     valid at once (when known): {(v.delay == 0).mean():.0%};  "
                  f"median wait before valid: {int(v.delay.median())} candles;  "
                  f"touched earlier while on the wrong side (ignored): {int(v.early_touch.sum())}")
        if len(t):
            print(f"     0.62-0.79 band at the touch: inside {(t.ote == 'inside').mean():.0%}, "
                  f"overlaps {(t.ote == 'overlaps').mean():.0%}, none {(t.ote == 'none').mean():.0%}")
        n_alt = int(((alt.kind == kind) & (alt.status == "touched")).sum())
        word = "expire at the next BOS/MSS" if EXPIRE_ON_NEXT_EVENT else "persist until their leg dies"
        other = "persisted until their leg dies" if EXPIRE_ON_NEXT_EVENT else "expired at the next BOS/MSS"
        print(f"     entered: {len(t)} when POIs {word}  vs  {n_alt} if they {other}")

    good = pois[pois.status == "touched"].sort_values("touch_idx")
    never = pois[pois.status == "never_valid"].sort_values("idx")
    os.makedirs(outdir, exist_ok=True)
    jobs = [("valid", r) for r in _spread(good, VALID_SAMPLES).itertuples(index=False)] + \
           [("never", r) for r in _spread(never, NEVER_SAMPLES).itertuples(index=False)]
    k = 0
    for tag, p in jobs:
        if tag == "never":
            vt = fib["leg_valid_to"].iloc[int(p.event_idx)]
            if vt != vt:
                continue
            snap = int(vt)
            f = fib.iloc[snap]
            fb = dict(start=f.start, end=f.end, start_idx=f.start_idx, end_idx=f.end_idx, mid=f.mid,
                      lvl62=f.lvl62, lvl79=f.lvl79, ote_lo=f.ote_lo, ote_hi=f.ote_hi)
            x_to = snap
            e = min(len(df), snap + 20)
            head = "NEVER on the right side of 0.5 (fib shown at the end of its leg)"
        else:
            fb = dict(start=p.fib_start, end=p.fib_end, start_idx=p.fib_start_idx, end_idx=p.fib_end_idx,
                      mid=p.mid, lvl62=p.lvl62, lvl79=p.lvl79, ote_lo=p.ote_lo, ote_hi=p.ote_hi)
            x_to = int(p.touch_idx)
            e = min(len(df), x_to + 40)
            head = (f"valid after {int(p.delay)} candles | entry {p.entry:.2f}, SL ref {p.sl_ref:.2f} | "
                    f"0.62-0.79: {p.ote}")
        s = max(0, min(int(p.idx), int(fb["start_idx"])) - CONTEXT)
        if e - s > MAX_CANDLES:
            s = e - MAX_CANDLES
        if e - s < MIN_CANDLES:
            s = max(0, e - MIN_CANDLES)
            e = min(len(df), s + MIN_CANDLES)
        fibs = [dict(x_start=int(fb["start_idx"]), x_end=int(fb["end_idx"]), x_to=x_to, start=fb["start"],
                     end=fb["end"], mid=fb["mid"], lvl62=fb["lvl62"], lvl79=fb["lvl79"],
                     ote_lo=fb["ote_lo"], ote_hi=fb["ote_hi"])]
        k += 1
        path = f"{outdir}/{pair}_{tf}_{k}_{tag}_{p.kind}_{pd.Timestamp(df['time'].iloc[int(p.idx)]):%Y%m%d_%H%M}.png"
        plot_structure(df, sw, ev, fv, s, e, path, f"{pair} {tf} {p.direction} {p.kind} | {head}",
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