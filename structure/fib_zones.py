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