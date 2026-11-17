"""Confluence scan on XAUUSD (H1 -> M15): trade table, funnel, H1 + M15 charts per leg, no-lookahead test."""
import os
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from broker.mt5 import MT5Broker
from config.settings import load_settings
from data.loader import fetch_tf
from strategy.confluence import prepare, scan, close_times
from structure.swing_methods import SWING_METHOD
from tools.plot_signal import plot_entry_set

PAIR = "XAUUSD"
DAYS = {"H1": 730, "M15": 365}
KEY = ["time", "direction", "entry", "sl", "tp", "kind", "seq"]


def build(raw):
    return {tf: prepare(df, tf, full=(tf == "M15")) for tf, df in raw.items()}


def cut(raw, i_end):
    t = raw["M15"]["time"].to_numpy()[i_end]
    return {"M15": raw["M15"].iloc[:i_end + 1], "H1": raw["H1"][close_times(raw["H1"], "H1") <= t]}


if __name__ == "__main__":
    pair = load_settings().pairs[PAIR]
    broker = MT5Broker()
    broker.connect()
    try:
        end = datetime.now(timezone.utc).replace(tzinfo=None)
        raw = {tf: fetch_tf(broker, pair.broker_symbol, tf, end - timedelta(days=d), end).reset_index(drop=True)
               for tf, d in DAYS.items()}
    finally:
        broker.disconnect()

    P = build(raw)
    s = scan(P, pair)
    print("\nswing methods:", SWING_METHOD)
    print("funnel:", s.attrs["stats"])
    if s.empty:
        raise SystemExit("no trades")
    print(f"trades: {len(s)}  (bull {int((s.direction == 'bull').sum())}, bear {int((s.direction == 'bear').sum())})"
          f" | outcomes {s.outcome.value_counts(dropna=False).to_dict()} | median RR {s.rr.median():.2f}")
    print("POI kind:", s.kind.value_counts().to_dict(), "| entry mode:", s.entry_mode.value_counts().to_dict(),
          "| attempt # in M15 leg:", s.seq.value_counts().sort_index().to_dict())

    # self-check: did any trade come after a close beyond the fib start (counted from the leg's end swing)
    # or beyond the touched zone's far edge (counted from the zone touch)?
    cl = P["M15"]["df"]["close"].to_numpy()
    tm = P["M15"]["df"]["time"].to_numpy()
    t1 = P["H1"]["df"]["time"].to_numpy()
    bad_start = bad_zone = 0
    for r in s.itertuples():
        seg = cl[int(np.searchsorted(tm, t1[int(r.fib_end_idx)])):int(r.idx)]
        zseg = cl[int(r.armed_idx):int(r.idx)]
        if r.direction == "bull":
            bad_start += bool(len(seg) and seg.min() < r.fib_start)
            bad_zone += bool(len(zseg) and zseg.min() < r.zone_bottom)
        else:
            bad_start += bool(len(seg) and seg.max() > r.fib_start)
            bad_zone += bool(len(zseg) and zseg.max() > r.zone_top)
    print(f"self-check: trades after a close beyond the fib start: {bad_start} (must be 0) | "
          f"beyond the zone's far edge: {bad_zone}")

    print(s[["time", "direction", "entry", "sl", "tp", "sl_pips", "tp_pips", "rr", "kind", "entry_mode", "seq",
             "outcome", "approach_rng_atr", "touch_rng_atr"]].head(20).to_string(float_format=lambda v: f"{v:.2f}"))

    os.makedirs("logs/confluence", exist_ok=True)
    for q, (_, g) in enumerate(s.groupby(["fib_leg", "m15_leg"], sort=True), 1):
        tag = f"{PAIR}_{q}_{g.direction.iloc[0]}_{pd.Timestamp(g.time.iloc[0]):%Y%m%d_%H%M}"
        for path in plot_entry_set(P, g.reset_index(drop=True), "logs/confluence", tag):
            print("   chart:", path)

    for frac in (0.4, 0.7, 0.9):
        i_end = int(len(raw["M15"]) * frac)
        c = scan(build(cut(raw, i_end)), pair)
        f = s[s.idx <= i_end].reset_index(drop=True)[KEY]
        ok = f.equals(c[KEY].reset_index(drop=True)) if len(c) else len(f) == 0
        print(f"no-lookahead cut at {frac:.0%}: full {len(f)} vs cut {len(c)} -> {'OK' if ok else 'MISMATCH'}")