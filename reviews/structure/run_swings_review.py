"""
Review of structure/swings.py (find_swings, fractal swings) on XAUUSD: self-checks, statistics, a table over n, charts.

Run from the project root:
    py -m reviews.structure.run_fractal_swings_review                    (M15, n=5)
    py -m reviews.structure.run_fractal_swings_review --tf H1 --n 5 --days 730
Offline, on a candle CSV (columns time, open, high, low, close):
    py -m reviews.structure.run_fractal_swings_review --csv candles.csv --pip 0.1

Only the raw fractal swings are judged here (no alternation step). Charts are drawn with tools/plot.py and go to
logs/fractal_swings/. An orange vertical line marks a swing that repeats the kind of the swing before it
(the ones an alternation step would have to drop).
"""
import argparse
import glob
import os
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from structure.swings import find_swings
from tools.plot import plot_structure

PAIR = "XAUUSD"
TF = "M15"
DAYS = 365
N = 5
SWEEP_N = (2, 3, 4, 5, 6, 8, 10, 12)
CHARTS = 8              # evenly spaced across the history
SHOW = 150              # candles per chart
CUTS = 6                # truncation points for the no-lookahead test
OUTDIR = "logs/fractal_swings"


# ---------------------------------------------------------------- data
def load_mt5(tf, days):
    from broker.mt5 import MT5Broker
    from config.settings import load_settings
    from data.loader import fetch_tf

    pair = load_settings().pairs[PAIR]
    broker = MT5Broker()
    broker.connect()
    try:
        end = datetime.now(timezone.utc).replace(tzinfo=None)
        df = fetch_tf(broker, pair.broker_symbol, tf, end - timedelta(days=days), end)
    finally:
        broker.disconnect()
    return df.iloc[:-1].reset_index(drop=True), pair.pip_size    # last candle may still be forming


def load_csv(path):
    return pd.read_csv(path, parse_dates=["time"]).reset_index(drop=True)


# ---------------------------------------------------------------- checks
def reference_swings(df, n):
    """The definition written a second way (rolling windows) to compare against find_swings."""
    h, l = df["high"], df["low"]
    prev_hi = h.rolling(n).max().shift(1)
    next_hi = h[::-1].rolling(n).max()[::-1].shift(-1)
    prev_lo = l.rolling(n).min().shift(1)
    next_lo = l[::-1].rolling(n).min()[::-1].shift(-1)
    hi = (h > prev_hi) & (h >= next_hi)          # comparisons with NaN (edges) are False
    lo = (l < prev_lo) & (l <= next_lo)
    return ({(int(i), "high") for i in np.flatnonzero(hi.to_numpy())}
            | {(int(i), "low") for i in np.flatnonzero(lo.to_numpy())})


def run_checks(df, sw, n):
    found = set(zip(sw["idx"].astype(int), sw["kind"]))
    ref = reference_swings(df, n)
    on_candle = all(
        abs(r.price - (df["high"].iat[r.idx] if r.kind == "high" else df["low"].iat[r.idx])) < 1e-9
        for r in sw.itertuples())
    order = list(zip(sw["confirmed_idx"], sw["idx"]))

    cuts_ok = True
    for c in np.linspace(len(df) * 0.3, len(df) - 1, CUTS).astype(int):
        part = find_swings(df.iloc[:c + 1], n)
        known = sw[sw["confirmed_idx"] <= c].reset_index(drop=True)
        cuts_ok &= part.equals(known)

    return [
        (f"same swings as the definition written a second way ({len(found)} vs {len(ref)})", found == ref),
        (f"confirmed_idx is idx + {n} for every swing", bool((sw["confirmed_idx"] == sw["idx"] + n).all())),
        ("swing price is that candle's high or low", on_candle),
        ("rows ordered by confirmation, then candle", order == sorted(order)),
        (f"no lookahead ({CUTS} cuts: truncated run == full run)", bool(cuts_ok)),
    ]


# ---------------------------------------------------------------- statistics
def repeat_mask(sw):
    """True for a swing whose kind equals the kind of the swing before it (in the order returned)."""
    k = sw["kind"].to_numpy()
    return np.concatenate([[False], k[1:] == k[:-1]])


def summarize(df, sw, n, pip, days, bar_min):
    if len(sw) < 3:
        return None
    price = sw["price"].to_numpy(float)
    idx = sw["idx"].to_numpy()
    kind = sw["kind"].to_numpy()
    close = df["close"].to_numpy(float)
    sign = np.where(kind == "high", 1, -1)
    giveback = (price - close[sw["confirmed_idx"].to_numpy()]) * sign     # price handed back when the swing is known
    opp = (kind[1:] != kind[:-1]) & (np.diff(idx) > 0)                    # neighbouring swings of opposite kind
    leg = np.abs(np.diff(price))[opp]
    gb = giveback[1:][opp]
    both = pd.Series(idx).duplicated().sum()                              # candles that are a swing high AND a swing low
    return dict(
        swings=len(sw),
        per_day=len(sw) / days,
        repeat_pct=float(100 * repeat_mask(sw).mean()),
        both=int(both),
        leg_pips=float(np.median(leg) / pip) if len(leg) else float("nan"),
        gap_bars=float(np.median(np.diff(np.unique(idx)))),
        delay_min=n * bar_min,
        giveback_pct=float(100 * np.median(gb[leg > 0] / leg[leg > 0])) if (leg > 0).any() else float("nan"),
    )


# ---------------------------------------------------------------- charts (tools/plot.py)
def make_charts(df, sw, n, tf):
    for old in glob.glob(os.path.join(OUTDIR, "fractal_*.png")):
        os.remove(old)
    no_events = pd.DataFrame(columns=["idx", "broken_idx", "level", "type"])
    vlines = [(int(i), "repeat", "#ff8c00") for i in sw["idx"][repeat_mask(sw)]]
    first = n * 3
    starts = np.linspace(first, max(first, len(df) - SHOW), CHARTS).astype(int)
    for k, a in enumerate(starts, 1):
        plot_structure(df, sw, no_events, None, a, min(a + SHOW, len(df)),
                       os.path.join(OUTDIR, f"fractal_{k:02d}.png"),
                       f"{PAIR} {tf} fractal swings n={n}  (orange line = repeats the previous swing's kind)",
                       vlines=vlines)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", help="candle CSV instead of MT5 (columns time, open, high, low, close)")
    ap.add_argument("--pip", type=float, default=0.1, help="pip size, only used with --csv")
    ap.add_argument("--tf", default=TF, help="timeframe when reading from MT5")
    ap.add_argument("--days", type=int, default=DAYS, help="history length when reading from MT5")
    ap.add_argument("--n", type=int, default=N)
    args = ap.parse_args()

    df, pip = (load_csv(args.csv), args.pip) if args.csv else load_mt5(args.tf, args.days)
    tf = args.tf
    days = (df["time"].iat[-1] - df["time"].iat[0]).total_seconds() / 86400
    bar_min = df["time"].diff().median().total_seconds() / 60
    print(f"fractal swings review: {PAIR} {tf}, {len(df)} candles, {df['time'].iat[0]:%Y-%m-%d} to "
          f"{df['time'].iat[-1]:%Y-%m-%d} ({days:.0f} days), pip size {pip}")
    print(f"selected: n={args.n}")

    sw = find_swings(df, args.n)

    print("\nChecks")
    for name, ok in run_checks(df, sw, args.n):
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")

    s = summarize(df, sw, args.n, pip, days, bar_min)
    print("\nStatistics (medians where it says so)")
    if s is None:
        print("  too few swings to summarize")
    else:
        print(f"  swings                           {s['swings']}  ({s['per_day']:.1f} per day)")
        print(f"  repeat the previous swing's kind {s['repeat_pct']:.0f}%")
        print(f"  candles that are high AND low    {s['both']}")
        print(f"  leg, opposite neighbours         {s['leg_pips']:.0f} pips")
        print(f"  candles between swing candles    {s['gap_bars']:.0f}")
        print(f"  delay until a swing is known     {args.n} candles ({s['delay_min']:.0f} min)")
        print(f"  price given back by then         {s['giveback_pct']:.0f}% of the leg")

    rows = []
    for n in SWEEP_N:
        m = summarize(df, find_swings(df, n), n, pip, days, bar_min)
        if m:
            rows.append(dict(sel="*" if n == args.n else "", n=n, swings=m["swings"],
                             per_day=round(m["per_day"], 1), repeat_pct=round(m["repeat_pct"]),
                             leg_pips=round(m["leg_pips"]), gap_bars=round(m["gap_bars"]),
                             delay_min=round(m["delay_min"]), giveback_pct=round(m["giveback_pct"])))
    pd.set_option("display.width", 200)
    print("\nTable over n (* = selected; medians; giveback = share of the leg already handed back when the swing is known)")
    print(pd.DataFrame(rows).to_string(index=False))

    make_charts(df, sw, args.n, tf)
    print(f"\n{CHARTS} charts saved to {OUTDIR}/ (fractal_01.png ... fractal_{CHARTS:02d}.png)")


if __name__ == "__main__":
    main()