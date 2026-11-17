"""
Explains what the confluence engine did around given times.
Usage:  py -m reviews.strategytrace_setup.py                       (the setups we have discussed)
"""
import sys
from datetime import datetime, timedelta, timezone

import pandas as pd

from broker.mt5 import MT5Broker
from config.settings import load_settings
from data.loader import fetch_tf
from strategy.confluence import scan, PIP, MAX_POI_PIPS
from structure.utils import atr
from reviews.strategy.run_confluence_review import build, PAIR, DAYS

DEFAULT = ["2025-12-26 10:45", "2026-02-25 12:15", "2026-10-05 02:45", "2026-09-10 11:30"]
HOURS_BEFORE, HOURS_AFTER = 60, 3

if __name__ == "__main__":
    targets = sys.argv[1:] or DEFAULT
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
    log = []
    scan(P, log=log)
    log.sort(key=lambda x: x[0])
    print(f"\nM15 ATR median: {atr(P['M15']['df']).median() / PIP:.0f} pips (POIs over {MAX_POI_PIPS} pips are not traded)")
    for t in targets:
        t0 = pd.Timestamp(t)
        print(f"\n=== around {t0} (last {HOURS_BEFORE}h before, {HOURS_AFTER}h after) ===")
        shown = 0
        for ts, msg in log:
            ts = pd.Timestamp(ts)
            if t0 - pd.Timedelta(hours=HOURS_BEFORE) <= ts <= t0 + pd.Timedelta(hours=HOURS_AFTER):
                print(f"{ts:%m-%d %H:%M}  {msg}")
                shown += 1
        if not shown:
            print("nothing logged in this window")