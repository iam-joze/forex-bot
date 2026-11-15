"""
Compare our structure layer with the smartmoneyconcepts library on XAUUSD (H4, H1, M15).
BOS/CHoCH and OB are run through the library on OUR swings, so only the detection logic differs.
The charts show the library as-is (its own swings) next to ours.
"""
import os
os.environ["SMC_CREDIT"] = "0"

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
from smartmoneyconcepts import smc

from broker.mt5 import MT5Broker
from config.settings import load_settings
from data.loader import fetch_tf
from structure.swings import find_swings, alternate
from structure.choch_mss import find_structure
from structure.fvg import find_fvgs, validate_fvgs
from structure.order_blocks import find_order_blocks
from structure.utils import atr
from tools.plot import plot_structure

PAIR = "XAUUSD"
DAYS = {"H4": 1095, "H1": 730, "M15": 365}
SWING_N = 5            # ours: fractal n=5 + alternation
LIB_SWING = 5          # library swing_length (window about the same width)
FVG_MIN_ATR, MIN_LEG_ATR = 0.3, 2.0
SAMPLES, WINDOW = 3, 200
OUT = "logs/smc"


# ---------- converters: library output <-> our table formats ----------
def ohlcv(df):
    d = df[["open", "high", "low", "close"]].astype(float).copy()
    v = "volume" if "volume" in df.columns else ("tick_volume" if "tick_volume" in df.columns else None)
    d["volume"] = df[v].astype(float) if v else 1.0
    return d.reset_index(drop=True)


def to_lib_swings(sw, n):
    ls = pd.DataFrame({"HighLow": np.nan, "Level": np.nan}, index=pd.RangeIndex(n))
    ls.loc[sw["idx"].to_numpy(), "HighLow"] = np.where(sw["kind"] == "high", 1.0, -1.0)
    ls.loc[sw["idx"].to_numpy(), "Level"] = sw["price"].to_numpy()
    return ls


def from_lib_swings(ls):
    m = ls["HighLow"].notna()
    return pd.DataFrame({"idx": ls.index[m].to_numpy(),
                         "kind": np.where(ls.loc[m, "HighLow"] > 0, "high", "low"),
                         "price": ls.loc[m, "Level"].to_numpy()})


def from_lib_events(bc):
    ok = (bc["BOS"].notna() | bc["CHOCH"].notna()) & (bc["BrokenIndex"] > 0)
    b = bc[ok]
    is_bos = b["BOS"].notna().to_numpy()
    val = np.where(is_bos, b["BOS"], b["CHOCH"])
    return pd.DataFrame({"idx": b["BrokenIndex"].astype(int).to_numpy(), "broken_idx": b.index.to_numpy(),
                         "level": b["Level"].to_numpy(), "type": np.where(is_bos, "BOS", "CHoCH"),
                         "direction": np.where(val > 0, "bull", "bear")})


def from_lib_fvg(lf):
    r = lf[lf["FVG"].notna()]
    mit = r["MitigatedIndex"].where(r["MitigatedIndex"] > 0)
    return pd.DataFrame({"c2": r.index.to_numpy(), "c1": r.index.to_numpy() - 1,
                         "direction": np.where(r["FVG"] > 0, "bull", "bear"),
                         "bottom": r["Bottom"].to_numpy(), "top": r["Top"].to_numpy(),
                         "mitigated_idx": mit.to_numpy(), "status": np.where(mit.isna(), "active", "mitigated")})


def from_lib_ob(lo):
    r = lo[lo["OB"].notna()]
    mit = (r["MitigatedIndex"].where(r["MitigatedIndex"] > 0) if "MitigatedIndex" in r.columns
           else pd.Series(np.nan, index=r.index))
    return pd.DataFrame({"idx": r.index.to_numpy(), "direction": np.where(r["OB"] > 0, "bull", "bear"),
                         "low": r["Bottom"].to_numpy(), "high": r["Top"].to_numpy(),
                         "mitigated_idx": mit.to_numpy(), "liq_grab": False})


# ---------- matching ----------
def near(a, b, tol):
    """How many values of a have a value of b within tol candles."""
    a, b = np.sort(np.asarray(a, float)), np.sort(np.asarray(b, float))
    if not len(a) or not len(b):
        return 0
    p = np.searchsorted(b, a)
    d = np.minimum(np.abs(a - b[np.clip(p, 0, len(b) - 1)]), np.abs(a - b[np.clip(p - 1, 0, len(b) - 1)]))
    return int((d <= tol).sum())


def keyset(df, col, extra):
    return set(zip(df[col].astype(int), df[extra]))


def cmp_swings(a, b):
    ex = len(keyset(a, "idx", "kind") & keyset(b, "idx", "kind"))
    a2 = sum(near(a.loc[a.kind == k, "idx"], b.loc[b.kind == k, "idx"], 2) for k in ("high", "low"))
    b2 = sum(near(b.loc[b.kind == k, "idx"], a.loc[a.kind == k, "idx"], 2) for k in ("high", "low"))
    print(f"  swings   ours {len(a)} | lib {len(b)} | exact {ex} | within 2 candles: ours {a2}/{len(a)}, lib {b2}/{len(b)}")


def cmp_events(a, b):
    both = keyset(a, "idx", "direction") & keyset(b, "idx", "direction")
    ta = {(int(r.idx), r.direction): ("CHoCH" if r.type == "MSS" else "BOS") for r in a.itertuples()}
    tb = {(int(r.idx), r.direction): r.type for r in b.itertuples()}
    same_type = sum(ta[k] == tb[k] for k in both)
    print(f"  BOS/MSS  ours {len(a)} (BOS {int((a.type == 'BOS').sum())}, MSS {int((a.type == 'MSS').sum())}) | "
          f"lib {len(b)} (BOS {int((b.type == 'BOS').sum())}, CHoCH {int((b.type == 'CHoCH').sum())}) | "
          f"same break candle+direction {len(both)} | same type among those {same_type}")


def cmp_fvg(a, b, df, A):
    kb = {(int(r.c2), r.direction): r for r in b.itertuples()}

    def ka_of(o):
        return {(int(r.c2) + o, r.direction): r for r in a.itertuples()}

    off = max((0, 1, -1), key=lambda o: len(ka_of(o).keys() & kb.keys()))
    ka = ka_of(off)
    both = ka.keys() & kb.keys()
    lib_only = [kb[k] for k in kb.keys() - ka.keys()]
    our_only = list(ka.keys() - kb.keys())
    small = sum(((r.top - r.bottom) / A[int(r.c2)]) < FVG_MIN_ATR for r in lib_only if A[int(r.c2)] == A[int(r.c2)])
    o, c = df["open"].to_numpy(), df["close"].to_numpy()
    wrong_col = sum(not ((c[k[0]] > o[k[0]]) if k[1] == "bull" else (c[k[0]] < o[k[0]])) for k in our_only)
    same = later = earlier = 0
    for k in both:
        ma, mb = ka[k].mitigated_idx, kb[k].mitigated_idx
        if pd.isna(ma) and pd.isna(mb):
            same += 1
        elif pd.isna(ma) or (not pd.isna(mb) and ma > mb):
            later += 1
        elif pd.isna(mb) or ma < mb:
            earlier += 1
        else:
            same += 1
    print(f"  FVG      ours {len(a)} | lib {len(b)} | same middle candle+direction {len(both)}"
          f"{'' if off == 0 else f' (our c2 is offset by {off} from the library middle candle)'}")
    print(f"           lib-only {len(lib_only)}: {small} smaller than {FVG_MIN_ATR} ATR, "
          f"{len(lib_only) - small} not on an impulse leg / no BOS-MSS")
    print(f"           ours-only {len(our_only)}: {wrong_col} where the middle candle has the wrong colour for the library")
    print(f"           mitigation on matched: same {same}, ours later/never {later}, ours earlier {earlier}")


def cmp_ob(a, b):
    ka, kb = keyset(a, "idx", "direction"), keyset(b, "idx", "direction")
    both = ka & kb
    da = {(int(r.idx), r.direction): r for r in a.itertuples()}
    db = {(int(r.idx), r.direction): r for r in b.itertuples()}
    same_zone = sum(abs(da[k].low - db[k].low) < 1e-6 and abs(da[k].high - db[k].high) < 1e-6 for k in both)
    a2 = sum(near(a.loc[a.direction == d, "idx"], b.loc[b.direction == d, "idx"], 2) for d in ("bull", "bear"))
    b2 = sum(near(b.loc[b.direction == d, "idx"], a.loc[a.direction == d, "idx"], 2) for d in ("bull", "bear"))
    print(f"  OB       ours {len(a)} | lib {len(b)} | same candle+direction {len(both)} (same zone {same_zone}) | "
          f"within 2 candles: ours {a2}/{len(a)}, lib {b2}/{len(b)}")


# ---------- one timeframe ----------
def review(df, tf):
    df = df.reset_index(drop=True)
    n = len(df)
    sw = alternate(find_swings(df, SWING_N))
    ev = find_structure(df, sw)
    fv = validate_fvgs(find_fvgs(df, FVG_MIN_ATR), ev, df)
    ob = find_order_blocks(df, ev, min_leg_atr=MIN_LEG_ATR)
    A = atr(df).to_numpy()

    o = ohlcv(df)
    ls_own = smc.swing_highs_lows(o, swing_length=LIB_SWING)
    ls_ours = to_lib_swings(sw, n)
    lib_sw = from_lib_swings(ls_own)
    lib_fv = from_lib_fvg(smc.fvg(o))
    ev_on_ours = from_lib_events(smc.bos_choch(o, ls_ours, close_break=True))
    ob_on_ours = from_lib_ob(smc.ob(o, ls_ours, close_mitigation=False))
    ev_own = from_lib_events(smc.bos_choch(o, ls_own, close_break=True))
    ob_own = from_lib_ob(smc.ob(o, ls_own, close_mitigation=False))

    print(f"\n=== {PAIR} {tf}: {n} candles, {df['time'].iloc[0]:%Y-%m-%d} -> {df['time'].iloc[-1]:%Y-%m-%d} ===")
    cmp_swings(sw, lib_sw)
    print("  -- library run on OUR swings (logic differences only) --")
    cmp_events(ev, ev_on_ours)
    cmp_ob(ob, ob_on_ours)
    cmp_fvg(fv, lib_fv, df, A)
    print(f"  -- library as-is (own swings): swings {len(lib_sw)}, BOS/CHoCH {len(ev_own)}, OB {len(ob_own)}, "
          f"FVG {len(lib_fv)}")

    os.makedirs(OUT, exist_ok=True)
    for k, c in enumerate((np.linspace(0.15, 0.85, SAMPLES) * n).astype(int), 1):
        e = min(n, max(0, c - WINDOW // 2) + WINDOW)
        s = max(0, e - WINDOW)
        base = f"{OUT}/{PAIR}_{tf}_{k}_{df['time'].iloc[s]:%Y%m%d_%H%M}"
        plot_structure(df, sw, ev, fv, s, e, base + "_ours.png",
                       f"{PAIR} {tf} OURS (swings n={SWING_N}, BOS/MSS, FVG, OB)", obs=ob)
        plot_structure(df, lib_sw, ev_own, lib_fv, s, e, base + "_smc.png",
                       f"{PAIR} {tf} SMC LIBRARY as-is (swing_length={LIB_SWING})", obs=ob_own)
        print("   charts:", base + "_ours.png", "|", base + "_smc.png")


if __name__ == "__main__":
    pair = load_settings().pairs[PAIR]
    broker = MT5Broker()
    broker.connect()
    try:
        end = datetime.now(timezone.utc).replace(tzinfo=None)
        data = {tf: fetch_tf(broker, pair.broker_symbol, tf, end - timedelta(days=d), end)
                for tf, d in DAYS.items()}
    finally:
        broker.disconnect()
    for tf, df in data.items():
        review(df, tf)