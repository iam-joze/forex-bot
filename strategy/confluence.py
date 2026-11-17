"""
confluence.py - H1 swing-leg fib -> H1 zone POI -> M15 MSS -> M15 POI entry -> position.
Causal: an H1 candle counts only once closed (close time = next candle's open).
One position at a time: the next POI is taken only after the previous trade hit its stop;
a take-profit ends the setup. D1 and H4 are not used.
Swing detector per timeframe: see structure/swing_methods.py.
scan(P, pair, rules=DEFAULT_RULES, log=[]) appends (time, message) notes explaining what the engine did
(see trace_setup.py). Pip size comes from the pair, entry/stop/simulation rules from risk/rules.py.
"""

import numpy as np
import pandas as pd
from structure.swing_methods import swings_for
from structure.choch_mss import find_structure
from structure.fvg import find_fvgs, validate_fvgs
from structure.order_blocks import find_order_blocks
from structure.fib_zones import provisional_fib, validate_pois
from structure.utils import atr
from risk.rules import DEFAULT_RULES, poi_entry, stop_loss, simulate

DUR = {"H1": np.timedelta64(1, "h"), "M15": np.timedelta64(15, "m")}
OTE = (0.62, 0.79)
MOM_ATR = 1.5          # FVGs and OBs are skipped when the approach candles are this big (ATR units)
ZONE_FAIL_KILLS = False  # True: an M15 close beyond the touched zone POI's far edge also ends the setup
EXCLUDE_MITIGATED = True  # True: a POI that was touched before it became valid counts as mitigated, not traded


def prepare(df, tf, full=False, fvg_min_atr=0.3, min_leg_atr=2.0):
    """Swings (method per timeframe), BOS/MSS events, FVGs, OBs.
    full=True (M15 only) also builds the BOS/MSS-leg fib and POI validity."""
    df = df.reset_index(drop=True)
    sw = swings_for(df, tf)
    ev = find_structure(df, sw)
    fv = validate_fvgs(find_fvgs(df, fvg_min_atr), ev, df)
    ob = find_order_blocks(df, ev, min_leg_atr=min_leg_atr)
    P = dict(df=df, sw=sw, ev=ev, fv=fv, ob=ob)
    if full:
        P["fib"] = provisional_fib(df, ev, sw)
        P["poi"] = validate_pois(df, P["fib"], ob, fv)
    return P


def close_times(df, tf):
    t = df["time"].to_numpy()
    return np.append(t[1:], t[-1] + DUR[tf])


def swing_leg_fib(df, sw, tol_atr=0.3, ote=OTE):
    """
    Fib of the CURRENT leg = the last two CONFIRMED alternating swings (no BOS/MSS needed).
    Latest confirmed swing is a high -> bull leg (low -> high); a low -> bear leg.
    Double top/bottom (within tol_atr x ATR): the first extreme is kept; the second one and the
    swing that led to it are ignored. Live from the candle the later swing is confirmed until the
    next swing is confirmed. leg_id = start_idx * 1e6 + end_idx.
    """
    n = len(df)
    a = atr(df).to_numpy()
    s = sw.sort_values("idx").reset_index(drop=True)
    sidx = s["idx"].to_numpy(int)
    px = s["price"].to_numpy(float)
    hi = (s["kind"] == "high").to_numpy()
    conf = s["confirmed_idx"].to_numpy(int)
    dirn = np.full(n, None, dtype=object)
    cols = {c: np.full(n, np.nan) for c in ("start", "end", "start_idx", "end_idx", "leg_id")}
    for j in range(1, len(s)):
        e = j
        tol = tol_atr * a[conf[j]]
        while e - 3 >= 0 and abs(px[e] - px[e - 2]) <= tol:
            e -= 2
        t0 = conf[j]
        t1 = (conf[j + 1] - 1) if j + 1 < len(s) else n - 1
        if t1 < t0:
            continue
        sl = slice(t0, t1 + 1)
        dirn[sl] = "bull" if hi[j] else "bear"
        cols["start"][sl], cols["end"][sl] = px[e - 1], px[e]
        cols["start_idx"][sl], cols["end_idx"][sl] = sidx[e - 1], sidx[e]
        cols["leg_id"][sl] = sidx[e - 1] * 1_000_000 + sidx[e]
    start, end = cols["start"], cols["end"]

    def price(r):
        return end + r * (start - end)

    p62, p79 = price(ote[0]), price(ote[1])
    return pd.DataFrame({"dir": dirn, **cols, "mid": price(0.5), "lvl62": p62, "lvl79": p79,
                         "ote_lo": np.minimum(p62, p79), "ote_hi": np.maximum(p62, p79)})


def pois_on_fib(Pp, fib, t_fib, k_of_poi):
    """
    OB/FVG of one timeframe (Pp) judged against the swing-leg fib of the same timeframe.
    k_of_poi[j] = fib-frame candle already closed when candle j of the POI frame closes.
    Valid from the first candle on which the POI was formed inside the leg, matches the leg direction
    and sits wholly on the right side of 0.5; valid until the leg is replaced.
    fail_idx = first close beyond the POI's far edge after it became valid.
    """
    mid, leg, st = fib["mid"].to_numpy(), fib["leg_id"].to_numpy(), fib["start_idx"].to_numpy()
    is_bull = (fib["dir"] == "bull").to_numpy()
    is_bear = (fib["dir"] == "bear").to_numpy()
    t_p = Pp["df"]["time"].to_numpy()
    c = Pp["df"]["close"].to_numpy()
    n = len(c)
    rows = []
    for kind, tbl in (("OB", Pp["ob"]), ("FVG", Pp["fv"])):
        if tbl is None or not len(tbl):
            continue
        for r in tbl.itertuples(index=False):
            bot, top, idx = (r.low, r.high, r.idx) if kind == "OB" else (r.bottom, r.top, r.c2)
            bull = r.direction == "bull"
            j0 = int(r.known_idx)
            kk = k_of_poi[j0:]
            kc = np.where(kk >= 0, kk, 0)
            side = (top <= mid[kc]) if bull else (bot >= mid[kc])
            formed = t_p[int(idx)] >= t_fib[np.nan_to_num(st[kc]).astype(int)]
            cond = (kk >= 0) & (is_bull[kc] if bull else is_bear[kc]) & side & formed
            if not cond.any():
                continue
            vf = j0 + int(np.argmax(cond))
            L = leg[kc[vf - j0]]
            ch = np.where(leg[kc[vf - j0:]] != L)[0]
            vt = vf + int(ch[0]) - 1 if len(ch) else n - 1
            seg = c[vf:vt + 1]
            bad = np.where(seg < bot)[0] if bull else np.where(seg > top)[0]
            rows.append((kind, int(idx), r.direction, bot, top, int(L), vf, vt,
                         vf + int(bad[0]) if len(bad) else np.nan))
    return pd.DataFrame(rows, columns=["kind", "idx", "direction", "bottom", "top", "leg",
                                       "valid_from", "valid_to", "fail_idx"])


def _active(g, k, bdir):
    if g is None:
        return []
    m = ((g.direction == bdir) & (g.valid_from <= k) & (g.valid_to >= k)
         & (g.fail_idx.isna() | (g.fail_idx > k)))
    return list(g[m].itertuples(index=False))


def scan(P, pair, rules=DEFAULT_RULES, log=None):
    """P = {'H1','M15': prepare(...)}; M15 must be prepared with full=True. Returns one row per trade.
    pair: PairConfig (its pip_size converts prices to pips). rules: risk.rules.ExecutionRules.
    log: optional list; (time, message) notes are appended to it."""
    pip = pair.pip_size
    df = P["M15"]["df"]
    tm, low, high, close = (df[c].to_numpy() for c in ("time", "low", "high", "close"))
    n, a = len(df), atr(df).to_numpy()
    d1 = P["H1"]["df"]
    t1 = d1["time"].to_numpy()
    k1 = np.searchsorted(close_times(d1, "H1"), tm, side="right") - 1

    def note(i, msg):
        if log is not None:
            log.append((tm[min(int(i), n - 1)], msg))

    fib = swing_leg_fib(d1, P["H1"]["sw"])
    f_dir = fib["dir"].to_numpy()
    fc = {c: fib[c].to_numpy() for c in ("leg_id", "start", "end", "mid", "ote_lo", "ote_hi",
                                         "start_idx", "end_idx")}
    g1 = {int(k): g for k, g in pois_on_fib(P["H1"], fib, t1, np.arange(len(d1))).groupby("leg")}

    st = dict(legs_followed=0, arms=0, trigger=0, bos_leg=0, opp_mss=0, high_broken=0,
              start_broken=0, zone_failed=0, momentum_skip=0, poi_over_max=0,
              poi_already_mitigated=0, trades=0)

    # M15 POIs: entry price and the candle on which it is reached (half-way for POIs taller than 40 pips)
    pm = P["M15"]["poi"]
    touch = {}
    for r in pm[pm.status == "touched"].itertuples(index=False):
        tag = f"{r.kind} {r.direction} {r.bottom:.2f}-{r.top:.2f}"
        if EXCLUDE_MITIGATED and bool(r.early_touch):              # touched before it became valid: mitigated
            st["poi_already_mitigated"] += 1
            note(r.touch_idx, f"POI {tag} not traded: already mitigated (touched before it was valid)")
            continue
        bull = r.direction == "bull"
        h_pips, entry, far, mode = poi_entry(r.bottom, r.top, bull, pip, rules)
        if entry is None:                                           # too big: not traded
            st["poi_over_max"] += 1
            note(r.touch_idx, f"POI {tag} not traded: {h_pips:.0f} pips tall (max {rules.max_poi_pips})")
            continue
        j0 = int(r.touch_idx)
        if mode == "half":
            seg = low[j0:int(r.valid_to) + 1] if bull else high[j0:int(r.valid_to) + 1]
            hit = np.where(seg <= entry)[0] if bull else np.where(seg >= entry)[0]
            if not len(hit):
                continue
            j0 += int(hit[0])
        touch.setdefault((int(r.event_idx), j0), []).append((r, entry, far, mode))

    ev = P["M15"]["ev"]
    e_idx, e_dir, e_typ = ev["idx"].to_numpy(), ev["direction"].to_numpy(), ev["type"].to_numpy()
    orig = dict(zip(ev["idx"].astype(int), ev["origin_idx"].astype(int)))

    def leg_tp(i, bull, entry, leg):
        """Take profit = extreme of the M15 leg since its origin, up to the previous candle
        (the recent swing high/low, confirmed or not). Must lie beyond the entry."""
        o = orig[leg]
        if i - o < 1:
            return np.nan
        ext = high[o:i].max() if bull else low[o:i].min()
        return ext if ((ext > entry) if bull else (ext < entry)) else np.nan

    sig, lg, armed, trig, cur, seq, min_i, ep, busy, last_sl, dead = \
        [], None, None, None, None, 0, 0, 0, 0, None, set()
    for i in range(1, n):
        k = k1[i]
        if armed is None:                       # follow the current H1 leg until its zone is touched
            L = fc["leg_id"][k] if k >= 0 else np.nan
            if L != L or int(L) in dead:
                lg = None
            elif lg is None or lg["id"] != int(L):
                cand = dict(id=int(L), dir=f_dir[k], start=fc["start"][k], end=fc["end"][k],
                            mid=fc["mid"][k], lo=fc["ote_lo"][k], hi=fc["ote_hi"][k],
                            s_idx=int(fc["start_idx"][k]), e_idx=int(fc["end_idx"][k]))
                # the end swing is only confirmed several candles after it formed: if price already closed
                # beyond the leg's start in that window, the leg is dead before it even starts
                seg = close[int(np.searchsorted(tm, t1[cand["e_idx"]])):i]
                if len(seg) and ((seg.min() < cand["start"]) if cand["dir"] == "bull" else (seg.max() > cand["start"])):
                    st["start_broken"] += 1
                    dead.add(cand["id"])
                    lg = None
                    note(i, f"{cand['dir']} H1 leg {cand['start']:.2f} -> {cand['end']:.2f} dead on arrival: "
                            f"price had already closed beyond its start")
                else:
                    lg = cand
                    min_i = i
                    st["legs_followed"] += 1
                    note(i, f"follow {cand['dir']} H1 leg {cand['start']:.2f} -> {cand['end']:.2f}")

        while ep < len(e_idx) and e_idx[ep] < i:                               # M15 events already closed
            if armed is not None:
                if trig is None:
                    if e_typ[ep] == "MSS" and e_dir[ep] == lg["dir"] and e_idx[ep] > armed["i"]:
                        trig = cur = int(e_idx[ep]); seq = 0; st["trigger"] += 1
                        note(i, f"M15 MSS trigger (broke on candle {int(e_idx[ep])})")
                elif e_dir[ep] == lg["dir"]:                                   # BOS in bias direction: new leg
                    cur = int(e_idx[ep]); st["bos_leg"] += 1
                    note(i, "new M15 BOS leg in the setup direction")
                else:                                                          # MSS the other way: reset zone
                    armed, trig, cur, min_i = None, None, None, i; st["opp_mss"] += 1
                    note(i, "zone reset: M15 MSS against the setup")
            ep += 1

        if lg is None:
            continue
        bdir, b = lg["dir"], (1 if lg["dir"] == "bull" else -1)

        # price cleared the fib leg (a close beyond its high, or beyond its start): the leg is dead,
        # whether or not its zone was touched yet
        j = i - 1
        beyond_end = close[j] > lg["end"] if b > 0 else close[j] < lg["end"]
        beyond_start = close[j] < lg["start"] if b > 0 else close[j] > lg["start"]
        zone_failed = (ZONE_FAIL_KILLS and armed is not None and
                       ((close[j] < armed["bot"]) if b > 0 else (close[j] > armed["top"])))
        if beyond_end or beyond_start or zone_failed:
            why = "high" if beyond_end else "start" if beyond_start else "zone far edge"
            st["high_broken" if beyond_end else "start_broken" if beyond_start else "zone_failed"] += 1
            note(i, f"{bdir} leg dead ({'armed' if armed else 'not armed'}): M15 close {close[j]:.2f} beyond its {why}")
            dead.add(lg["id"])
            armed = trig = cur = lg = None
            continue

        if armed is None and i >= min_i:
            for r in _active(g1.get(lg["id"]), k, bdir):
                if (low[i] <= r.top) if b > 0 else (high[i] >= r.bottom):
                    armed = dict(i=i, kind=r.kind, idx=int(r.idx), bot=r.bottom, top=r.top, k1=int(k))
                    st["arms"] += 1
                    note(i, f"ARMED: price touched H1 {r.kind} zone {r.bottom:.2f}-{r.top:.2f}")
                    break

        if trig is not None and cur is not None and i >= busy:
            mom = (max(high[i - 1] - low[i - 1], high[i - 2] - low[i - 2]) / a[i - 1]
                   if i >= 2 and a[i - 1] == a[i - 1] else 0.0)
            for r, entry, far, mode in sorted(touch.get((cur, i), []), key=lambda t: -t[1] if b > 0 else t[1]):
                if i == busy and not ((entry > last_sl) if b < 0 else (entry < last_sl)):
                    continue            # same candle as the stop: only POIs price had to pass the stop to reach
                if mom >= MOM_ATR:                                             # high-momentum approach: no FVGs/OBs
                    st["momentum_skip"] += 1
                    note(i, f"{r.kind} entry {entry:.2f} skipped: approach {mom:.2f} ATR >= {MOM_ATR}")
                    continue
                sl, dist = stop_loss(entry, far, b > 0, pip, rules)             # covers the whole POI
                tp = leg_tp(i, b > 0, entry, cur)
                x, out = simulate(low, high, i, b > 0, sl, tp)
                seq += 1
                st["trades"] += 1
                note(i, f"TRADE {bdir} {r.kind} entry {entry:.2f} SL {sl:.2f} TP {tp:.2f} -> {out}")
                tp_pips = abs(tp - entry) / pip if tp == tp else np.nan
                sig.append(dict(time=tm[i], idx=i, direction=bdir, entry=entry, sl=sl, tp=tp,
                                sl_pips=dist / pip, tp_pips=tp_pips, rr=tp_pips / (dist / pip),
                                kind=r.kind, entry_mode=mode, ote=r.ote, m15_leg=cur, seq=seq,
                                zone_kind=armed["kind"], zone_idx=armed["idx"], fib_leg=lg["id"],
                                fib_start=lg["start"], fib_end=lg["end"], fib_mid=lg["mid"],
                                fib_lo=lg["lo"], fib_hi=lg["hi"], fib_start_idx=lg["s_idx"],
                                fib_end_idx=lg["e_idx"], poi_idx=int(r.idx), armed_idx=armed["i"],
                                armed_k1=armed["k1"], trig=trig, zone_bottom=armed["bot"],
                                zone_top=armed["top"], approach_rng_atr=mom,
                                touch_rng_atr=(high[i] - low[i]) / a[i - 1] if a[i - 1] == a[i - 1] else np.nan,
                                exit_idx=x, outcome=out))
                busy, last_sl = (x if out else n), sl
                if out == "TP":                                                 # setup complete
                    dead.add(lg["id"])
                    armed = trig = cur = lg = None
                    break
                if busy != i:                                                   # still open / stopped later
                    break
                # stopped out inside this same candle: keep scanning POIs that lie beyond the stop
    out = pd.DataFrame(sig)
    out.attrs["stats"] = st
    return out