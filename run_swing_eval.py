"""
Scores swing-detection rules against YOUR hand-marked swings.
Reads calibration/swings_<TF>.csv  (columns: time_utc,kind  kind = high|low)
"""
from datetime import timedelta
import pandas as pd
from broker.mt5 import MT5Broker
from config.settings import load_settings
from structure.swings import find_swings, find_swings_zigzag, alternate

PAIR = "EURUSD"
TFS = ["H1", "M15"]          # add "H4", "D1" once those are anchored/marked
TOL = 2                      # candles of slack when matching a swing in time
WARMUP_DAYS = 12


def score(det: pd.DataFrame, mine: pd.DataFrame, df: pd.DataFrame, tol: int):
    times = pd.DatetimeIndex(df["time"])
    my = []
    for r in mine.itertuples(index=False):
        i = int(times.get_indexer([pd.Timestamp(r.time_utc)], method="nearest")[0])
        my.append((i, r.kind))
    lo, hi = min(i for i, _ in my) - tol, max(i for i, _ in my) + tol
    d = [(int(s.idx), s.kind) for s in det.itertuples(index=False) if lo <= s.idx <= hi]
    used, hits = set(), 0
    for i, k in my:
        for j, (di, dk) in enumerate(d):
            if j not in used and dk == k and abs(di - i) <= tol:
                used.add(j); hits += 1
                break
    rec = hits / len(my)
    prec = hits / len(d) if d else 0.0
    f1 = 2 * rec * prec / (rec + prec) if rec + prec else 0.0
    return rec, prec, f1, len(d), len(my)


settings = load_settings()
pair = settings.pairs[PAIR]
broker = MT5Broker()
broker.connect()
rows = []
try:
    for tf in TFS:
        mine = pd.read_csv(f"calibration/swings_{tf}.csv")
        t0 = pd.Timestamp(mine["time_utc"].min()).to_pydatetime() - timedelta(days=WARMUP_DAYS)
        t1 = pd.Timestamp(mine["time_utc"].max()).to_pydatetime() + timedelta(days=2)
        df = broker.get_candles_range(pair.broker_symbol, tf, t0, t1).reset_index(drop=True)
        cands = {}
        for n in (2, 3, 4, 5, 6, 8, 10, 12, 15):
            cands[f"fractal n={n} + alternate"] = alternate(find_swings(df, n))
        for m in (1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0):
            cands[f"zigzag {m} ATR"] = find_swings_zigzag(df, m)
        for name, det in cands.items():
            rec, prec, f1, nd, nm = score(det, mine, df, TOL)
            rows.append((tf, name, round(rec, 2), round(prec, 2), round(f1, 2), nd, nm))
finally:
    broker.disconnect()

res = pd.DataFrame(rows, columns=["tf", "rule", "recall", "precision", "f1", "detected", "yours"])
pd.set_option("display.width", 200)
for tf, g in res.groupby("tf", sort=False):
    print(f"\n=== {tf} (top 6 by F1) ===")
    print(g.sort_values("f1", ascending=False).head(6).drop(columns="tf").to_string(index=False))
    