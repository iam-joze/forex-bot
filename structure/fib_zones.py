import numpy as np
import pandas as pd

OTE = (0.62, 0.79)


def fib_state(df: pd.DataFrame, events: pd.DataFrame, swings: pd.DataFrame, ote=OTE,
              carry_over: bool = False) -> pd.DataFrame:
    """
    Fib of the leg of the LATEST BOS/MSS, anchored ONLY to marked swings (no lookahead).

    For a BOS/MSS event:
      start = the marked swing at the leg's origin: the lowest swing low (bullish) / highest swing
              high (bearish) between the broken swing and the break candle
      end   = the most extreme marked swing after `start` (highest swing high / lowest swing low),
              counting only swings already CONFIRMED on the candle in question
    The fib exists once both swings are confirmed (and the break has happened) and is replaced
    by the next BOS/MSS. Events whose origin is not a marked swing get no fib.
    Levels are retracements measured from `end` back toward `start`: mid = 0.5, plus the 0.62
    and 0.79 levels and the band between them (noted, not required).
    Bullish leg: below `mid` is discount. Bearish leg: above `mid` is premium.
    `draw_to_idx` = the next marked swing after `end` (hindsight, for drawing only).
    carry_over=True keeps the previous fib in force while the newest leg has no confirmed end
    swing yet (default False: no fib until the latest leg's own swings are marked).
    Columns: dir, start, end, start_idx, end_idx, leg_event_idx, mid, lvl62, lvl79,
             ote_lo, ote_hi, draw_to_idx.  df.attrs['stats'] = legs / legs_with_fib.
    """
    n = len(df)
    s_idx = swings["idx"].to_numpy()
    s_px = swings["price"].to_numpy(dtype=float)
    s_hi = (swings["kind"] == "high").to_numpy()
    s_conf = swings["confirmed_idx"].to_numpy()
    sorted_idx = np.sort(s_idx)

    dirn = np.full(n, None, dtype=object)
    start = np.full(n, np.nan)
    end = np.full(n, np.nan)
    start_idx = np.full(n, np.nan)
    end_idx = np.full(n, np.nan)
    leg_event = np.full(n, np.nan)
    draw_to = np.full(n, np.nan)

    ev = list(events.itertuples(index=False))
    with_fib = 0
    for k, e in enumerate(ev):
        vf = int(e.idx)
        vt = int(ev[k + 1].idx) - 1 if k + 1 < len(ev) else n - 1
        bull = e.direction == "bull"
        m = (s_hi == (not bull)) & (s_idx >= int(e.broken_idx)) & (s_idx <= vf)
        if not m.any():
            continue
        cand = np.where(m)[0]
        j = cand[np.argmin(s_px[cand])] if bull else cand[np.argmax(s_px[cand])]
        S_idx, S_px, S_conf = int(s_idx[j]), s_px[j], int(s_conf[j])

        me = (s_hi == bull) & (s_idx > S_idx) & (s_conf <= vt)
        if not me.any():
            continue
        ce = np.where(me)[0]
        ce = ce[np.argsort(s_conf[ce], kind="stable")]

        best = None
        assigned = False
        for q, jj in enumerate(ce):
            if best is None or (s_px[jj] > s_px[best] if bull else s_px[jj] < s_px[best]):
                best = jj
            t_from = max(int(s_conf[jj]), S_conf, vf)
            t_to = (max(int(s_conf[ce[q + 1]]), S_conf, vf) - 1) if q + 1 < len(ce) else vt
            if t_to < t_from:
                continue
            sl = slice(t_from, t_to + 1)
            dirn[sl] = e.direction
            start[sl] = S_px
            end[sl] = s_px[best]
            start_idx[sl] = S_idx
            end_idx[sl] = s_idx[best]
            leg_event[sl] = vf
            pos = np.searchsorted(sorted_idx, s_idx[best], side="right")
            draw_to[sl] = sorted_idx[pos] if pos < len(sorted_idx) else n - 1
            assigned = True
        with_fib += int(assigned)

    def price(r):                      # retracement r measured from `end` toward `start`
        return end + r * (start - end)

    p62, p79 = price(ote[0]), price(ote[1])
    out = pd.DataFrame({
        "dir": dirn, "start": start, "end": end, "start_idx": start_idx, "end_idx": end_idx,
        "leg_event_idx": leg_event, "mid": price(0.5), "lvl62": p62, "lvl79": p79,
        "ote_lo": np.minimum(p62, p79), "ote_hi": np.maximum(p62, p79), "draw_to_idx": draw_to,
    })
    if carry_over:
        out = out.ffill()
    out.attrs["stats"] = {"legs": len(ev), "legs_with_fib": with_fib}
    return out


def evaluate_pois(df: pd.DataFrame, fib: pd.DataFrame, obs: pd.DataFrame = None,
                  fvgs: pd.DataFrame = None) -> pd.DataFrame:
    """
    Judge every order block / validated FVG against the fib in force when price comes back to it.

    Evaluation candle = the candle BEFORE its first touch (mitigation), or the last candle
    if it is still untouched (never earlier than the candle it became known).
    If no fib exists yet on that candle (its end swing is not confirmed), side = 'no_fib'.
    A POI is on the right side when its direction matches the fib's trend AND
      bullish POI: its whole zone is in discount (top <= mid)
      bearish POI: its whole zone is in premium (bottom >= mid)
    `side` is 'discount' / 'premium' / 'straddles' relative to mid.
    ote (noted only): 'inside' / 'overlaps' / 'none' vs the 0.62-0.79 band.
    """
    n = len(df)
    f_dir = fib["dir"].to_numpy()
    names = ("start", "end", "start_idx", "end_idx", "mid", "lvl62", "lvl79", "ote_lo", "ote_hi", "draw_to_idx")
    cols = {c: fib[c].to_numpy() for c in names}
    nan = np.nan
    rows = []

    def add(kind, idx, direction, bottom, top, known, mit):
        mit = None if (mit is None or (isinstance(mit, float) and np.isnan(mit))) else int(mit)
        ev = (mit - 1) if mit is not None else n - 1
        ev = max(ev, int(known))
        if not isinstance(f_dir[ev], str):          # no fib in force on that candle
            rows.append((kind, int(idx), direction, bottom, top, int(known), mit, ev, None,
                         nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, "no_fib", False, False, "none"))
            return
        mid = cols["mid"][ev]
        side = "discount" if top <= mid else ("premium" if bottom >= mid else "straddles")
        match = f_dir[ev] == direction
        right = match and ((direction == "bull" and side == "discount") or
                           (direction == "bear" and side == "premium"))
        lo, hi = cols["ote_lo"][ev], cols["ote_hi"][ev]
        if bottom >= lo and top <= hi:
            ote = "inside"
        elif top >= lo and bottom <= hi:
            ote = "overlaps"
        else:
            ote = "none"
        rows.append((kind, int(idx), direction, bottom, top, int(known), mit, ev, f_dir[ev],
                     cols["start"][ev], cols["end"][ev], cols["start_idx"][ev], cols["end_idx"][ev],
                     mid, cols["lvl62"][ev], cols["lvl79"][ev], lo, hi, cols["draw_to_idx"][ev],
                     side, match, right, ote))

    if obs is not None and len(obs):
        for o in obs.itertuples(index=False):
            add("OB", o.idx, o.direction, o.low, o.high, o.known_idx, o.mitigated_idx)
    if fvgs is not None and len(fvgs):
        for f in fvgs.itertuples(index=False):
            add("FVG", f.c2, f.direction, f.bottom, f.top, f.known_idx, f.mitigated_idx)

    return pd.DataFrame(rows, columns=[
        "kind", "idx", "direction", "bottom", "top", "known_idx", "mitigated_idx", "eval_idx",
        "fib_dir", "fib_start", "fib_end", "fib_start_idx", "fib_end_idx", "mid", "lvl62", "lvl79",
        "ote_lo", "ote_hi", "fib_draw_to", "side", "trend_match", "right_side", "ote"])

# ---------------------------------------------------------------------------------------------
# Provisional fib: available from the break candle, no waiting for swings to be confirmed
# ---------------------------------------------------------------------------------------------
def provisional_fib(df: pd.DataFrame, events: pd.DataFrame, swings: pd.DataFrame,
                    double_top_tol_atr: float = 0.3, atr_period: int = 14, ote=OTE) -> pd.DataFrame:
    """
    The fib of the latest BOS/MSS leg, live from the break candle (no lookahead), anchored to
    MARKED swings as soon as they exist and only provisional until then.
      start = the marked swing at the leg's origin once it is confirmed (before that: the raw origin)
      end   = the marked extreme swing once confirmed (first of equal extremes within
              double_top_tol_atr x ATR, as in a double top), but if price has run beyond it by more than
              that tolerance, or no end swing is marked yet, the raw running extreme
    So the range and its 0.5 move with price during the leg, and snap onto marked swings.
    Replaced by the next BOS/MSS. Columns: dir, start, end, start_idx, end_idx, leg_event_idx,
    leg_valid_to, mid, lvl62, lvl79, ote_lo, ote_hi.
    """
    from structure.utils import atr
    n = len(df)
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    a = atr(df, atr_period).to_numpy()
    s_idx = swings["idx"].to_numpy()
    s_px = swings["price"].to_numpy(dtype=float)
    s_hi = (swings["kind"] == "high").to_numpy()
    s_conf = swings["confirmed_idx"].to_numpy()

    dirn = np.full(n, None, dtype=object)
    cols = {c: np.full(n, np.nan) for c in ("start", "end", "start_idx", "end_idx", "leg_event_idx", "leg_valid_to")}

    ev = list(events.itertuples(index=False))
    for k, e in enumerate(ev):
        vf = int(e.idx)
        vt = int(ev[k + 1].idx) - 1 if k + 1 < len(ev) else n - 1
        o = int(e.origin_idx)
        bull = e.direction == "bull"
        if vt < vf:
            continue

        # raw running extreme of the leg since its origin
        seg = high[o:vt + 1] if bull else low[o:vt + 1]
        run = np.maximum.accumulate(seg) if bull else np.minimum.accumulate(seg)
        pos = np.arange(len(seg))
        run_pos = np.maximum.accumulate(np.where((seg >= run) if bull else (seg <= run), pos, 0))

        # marked start swing (lowest swing low / highest swing high between the broken swing and the break)
        m = (s_hi == (not bull)) & (s_idx >= int(e.broken_idx)) & (s_idx <= vf)
        if m.any():
            cand = np.where(m)[0]
            j = cand[np.argmin(s_px[cand])] if bull else cand[np.argmax(s_px[cand])]
            S_px, S_idx, S_conf = s_px[j], int(s_idx[j]), int(s_conf[j])
        else:
            S_px, S_idx, S_conf = (low[o] if bull else high[o]), o, vf
        raw_S = low[o] if bull else high[o]

        # marked end swings, first of equal extremes wins
        me = (s_hi == bull) & (s_idx > S_idx) & (s_conf <= vt)
        ce = np.where(me)[0]
        ce = ce[np.argsort(s_conf[ce], kind="stable")]
        mk_px = np.full(vt - vf + 1, np.nan)
        mk_idx = np.full(vt - vf + 1, np.nan)
        best = None
        for jj in ce:
            tol_c = double_top_tol_atr * a[int(s_conf[jj])] if a[int(s_conf[jj])] == a[int(s_conf[jj])] else 0.0
            if best is None or (s_px[jj] > s_px[best] + tol_c if bull else s_px[jj] < s_px[best] - tol_c):
                best = jj
            t_from = max(int(s_conf[jj]), vf)
            if t_from <= vt:
                mk_px[t_from - vf:] = s_px[best]
                mk_idx[t_from - vf:] = s_idx[best]

        for t in range(vf, vt + 1):
            tol = double_top_tol_atr * a[t] if a[t] == a[t] else 0.0
            r_px, r_idx = run[t - o], o + run_pos[t - o]
            end_px, end_i = r_px, r_idx                      # provisional: raw running extreme
            mpx = mk_px[t - vf]
            if mpx == mpx and S_conf <= t and ((r_px <= mpx + tol) if bull else (r_px >= mpx - tol)):
                end_px, end_i = mpx, mk_idx[t - vf]          # snap onto the marked swing
            if S_conf <= t:
                st_px, st_i = S_px, S_idx
            else:
                st_px, st_i = raw_S, o
            dirn[t] = e.direction
            cols["start"][t], cols["end"][t] = st_px, end_px
            cols["start_idx"][t], cols["end_idx"][t] = st_i, end_i
            cols["leg_event_idx"][t], cols["leg_valid_to"][t] = vf, vt

    start, end = cols["start"], cols["end"]

    def price(r):
        return end + r * (start - end)

    p62, p79 = price(ote[0]), price(ote[1])
    return pd.DataFrame({
        "dir": dirn, **cols, "mid": price(0.5), "lvl62": p62, "lvl79": p79,
        "ote_lo": np.minimum(p62, p79), "ote_hi": np.maximum(p62, p79)})


POI_COLS = ["kind", "idx", "direction", "bottom", "top", "event_idx", "known_idx", "status",
            "valid_from", "valid_to", "delay", "early_touch", "touch_idx", "entry", "sl_ref", "ote",
            "snap_idx", "fib_start", "fib_end", "fib_start_idx", "fib_end_idx", "mid", "lvl62", "lvl79",
            "ote_lo", "ote_hi"]


def validate_pois(df: pd.DataFrame, fib: pd.DataFrame, obs: pd.DataFrame = None,
                  fvgs: pd.DataFrame = None, expire_on_next_event: bool = True) -> pd.DataFrame:
    """
    A POI belongs to the BOS/MSS leg that made it. It becomes VALID the first candle (from the
    moment it is known) on which its whole zone sits on the right side of that leg's CURRENT 0.5
    (bullish: top <= mid, bearish: bottom >= mid); this is judged inside the leg's own window.
    Once valid it stays valid ("ever sits") until the leg dies (a close beyond its origin) and,
    if expire_on_next_event=True, no later than the next BOS/MSS (default). With False it can be
    entered after later BOS/MSS events too, until the leg dies.
    Entry touch = first touch of the POI's entry edge after it became valid
      (bullish: low <= top, entry = top, sl_ref = bottom; bearish mirrored).
    status: 'touched' | 'untouched' (valid, not yet touched) | 'never_valid'.
    early_touch: it was touched BEFORE it became valid (that touch is ignored).
    ote (noted only): inside / overlaps / none vs the 0.62-0.79 band at the touch (or at validity).
    snap_idx / fib_*: the fib state used for the ote note and for drawing.
    """
    n = len(df)
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    close = df["close"].to_numpy()
    c = {k: fib[k].to_numpy() for k in ("start", "end", "start_idx", "end_idx", "mid", "lvl62",
                                        "lvl79", "ote_lo", "ote_hi", "leg_valid_to")}
    rows = []

    def add(kind, idx, direction, bottom, top, event_idx, known):
        bull = direction == "bull"
        e0 = int(event_idx)
        vt = c["leg_valid_to"][e0]
        base = (kind, int(idx), direction, bottom, top, e0, int(known))
        empty = (np.nan,) * 18
        if np.isnan(vt):
            rows.append(base + ("never_valid",) + empty)
            return
        vt = int(vt)
        start_px = c["start"][e0]
        seg_end = vt if expire_on_next_event else n - 1
        seg = close[e0 + 1:seg_end + 1]
        bad = np.where(seg < start_px)[0] if bull else np.where(seg > start_px)[0]
        touch_to = e0 + int(bad[0]) if len(bad) else seg_end      # leg dies on a close beyond its origin
        leg_to = min(vt, touch_to)                                  # validity is judged inside the leg's window
        t0 = int(known)
        if t0 > leg_to:
            rows.append(base + ("never_valid",) + empty)
            return
        mids = c["mid"][t0:leg_to + 1]
        cond = (top <= mids) if bull else (bottom >= mids)
        if not cond.any():
            rows.append(base + ("never_valid",) + empty)
            return
        vfrom = t0 + int(np.argmax(cond))
        rng = np.arange(t0 + 1, touch_to + 1)
        hit = (low[rng] <= top) if bull else (high[rng] >= bottom)
        touch = None
        early = False
        for j in rng[hit]:
            if j > vfrom:
                touch = int(j)
                break
            early = True
        snap = min(touch - 1, vt) if touch is not None else vfrom
        lo, hi = c["ote_lo"][snap], c["ote_hi"][snap]
        ote = "inside" if (bottom >= lo and top <= hi) else ("overlaps" if (top >= lo and bottom <= hi) else "none")
        rows.append(base + ("touched" if touch is not None else "untouched", vfrom, touch_to, vfrom - t0,
                            early, touch if touch is not None else np.nan,
                            top if bull else bottom, bottom if bull else top, ote, snap,
                            c["start"][snap], c["end"][snap], c["start_idx"][snap], c["end_idx"][snap],
                            c["mid"][snap], c["lvl62"][snap], c["lvl79"][snap], lo, hi))

    if obs is not None and len(obs):
        for o in obs.itertuples(index=False):
            add("OB", o.idx, o.direction, o.low, o.high, o.break_idx, o.known_idx)
    if fvgs is not None and len(fvgs):
        for f in fvgs.itertuples(index=False):
            add("FVG", f.c2, f.direction, f.bottom, f.top, f.break_idx, f.known_idx)
    return pd.DataFrame(rows, columns=POI_COLS)