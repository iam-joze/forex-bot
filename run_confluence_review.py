"""Confluence scan on XAUUSD (H1 -> M15): trade table, funnel, H1 + M15 charts per leg, no-lookahead test."""
import os
from datetime import datetime, timedelta, timezone

import pandas as pd

from broker.mt5 import MT5Broker
from config.settings import load_settings
from data.loader import fetch_tf
from strategy.confluence import prepare, scan, close_times
from tools.plot_signal import plot_entry_set

PAIR = "XAUUSD"
DAYS = {"H1": 730, "M15": 365}
KEY = ["time", "direction", "entry", "sl", "tp", "kind", "seq"]


def build(raw):
    return {tf: prepare(df, full=(tf == "M15")) for tf, df in raw.items()}


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
    s = scan(P)
    print("\nfunnel:", s.attrs["stats"])
    if s.empty:
        raise SystemExit("no trades")
    print(f"trades: {len(s)}  (bull {int((s.direction == 'bull').sum())}, bear {int((s.direction == 'bear').sum())})"
          f" | outcomes {s.outcome.value_counts(dropna=False).to_dict()} | median RR {s.rr.median():.2f}")
    print("POI kind:", s.kind.value_counts().to_dict(), "| entry mode:", s.entry_mode.value_counts().to_dict(),
          "| attempt # in M15 leg:", s.seq.value_counts().sort_index().to_dict())
    print(s[["time", "direction", "entry", "sl", "tp", "sl_pips", "tp_pips", "rr", "kind", "entry_mode", "seq",
              "outcome", "approach_rng_atr", "touch_rng_atr"]].head(20).to_string(float_format=lambda v: f"{v:.2f}"))

    os.makedirs("logs/confluence", exist_ok=True)
    for q, (_, g) in enumerate(s.groupby(["fib_leg", "m15_leg"], sort=True), 1):
        tag = f"{PAIR}_{q}_{g.direction.iloc[0]}_{pd.Timestamp(g.time.iloc[0]):%Y%m%d_%H%M}"
        for path in plot_entry_set(P, g.reset_index(drop=True), "logs/confluence", tag):
            print("   chart:", path)

    for frac in (0.4, 0.7, 0.9):
        i_end = int(len(raw["M15"]) * frac)
        c = scan(build(cut(raw, i_end)))
        f = s[s.idx <= i_end].reset_index(drop=True)[KEY]
        ok = f.equals(c[KEY].reset_index(drop=True)) if len(c) else len(f) == 0
        print(f"no-lookahead cut at {frac:.0%}: full {len(f)} vs cut {len(c)} -> {'OK' if ok else 'MISMATCH'}")