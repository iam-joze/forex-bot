import numpy as np
from tools.plot import plot_structure


def _window(lo, hi, n):
    s, e = max(0, int(lo)), min(n, int(hi))
    if e - s > 220:
        s = e - 220
    if e - s < 160:
        s = max(0, e - 160)
        e = min(n, s + 160)
    return s, e


def _fib(r0, to_x, x_to):
    st, en = r0.fib_start, r0.fib_end
    return [dict(x_start=to_x(int(r0.fib_start_idx)), x_end=to_x(int(r0.fib_end_idx)), x_to=x_to,
                 start=st, end=en, mid=r0.fib_mid, lvl62=en + 0.62 * (st - en), lvl79=en + 0.79 * (st - en),
                 ote_lo=r0.fib_lo, ote_hi=r0.fib_hi)]


def _zone_x0(r0):
    return int(r0.zone_idx) - 2 if r0.zone_kind == "FVG" else int(r0.zone_idx)


def _positions(g, n):
    return [dict(x0=int(r.idx), x1=max(int(r.exit_idx) if isinstance(r.outcome, str) else n, int(r.idx) + 3),
                 entry=r.entry, sl=r.sl, tp=r.tp, sl_pips=r.sl_pips, tp_pips=r.tp_pips,
                 outcome=r.outcome if isinstance(r.outcome, str) else None) for r in g.itertuples()]


def plot_entry_set(P, g, outdir, tag):
    """g = trades of one M15 leg. Draws H1 (fib + zone POI touched) and M15 (MSS + positions)."""
    r0 = g.iloc[0]
    d1, dm = P["H1"]["df"], P["M15"]["df"]
    t1, tm = d1["time"].to_numpy(), dm["time"].to_numpy()
    label = f"H1 {r0.zone_kind} zone"
    paths = []

    # --- H1: fib of the locked leg and the zone POI that was touched
    s, e = _window(min(int(r0.fib_start_idx), _zone_x0(r0)) - 15, int(r0.armed_k1) + 40, len(d1))
    path = f"{outdir}/{tag}_H1.png"
    plot_structure(d1, P["H1"]["sw"], P["H1"]["ev"], P["H1"]["fv"], s, e, path,
                   f"H1 {r0.direction} | fib of the current leg | zone: {label}",
                   obs=P["H1"]["ob"], fibs=_fib(r0, lambda i: i, int(r0.armed_k1) + 3),
                   highlights=[dict(x0=_zone_x0(r0), x1=int(r0.armed_k1) + 1, bottom=r0.zone_bottom,
                                    top=r0.zone_top, color="#ef6c00", label=label + " touched")],
                   vlines=[(int(r0.armed_k1), "zone touched", "#1976d2")])
    paths.append(path)

    # --- M15: MSS, zone, entries as positions
    to_m = lambda i: int(np.searchsorted(tm, t1[i]))
    last = int(g.idx.max())
    hi_end = max(last + 40, min(int(g.exit_idx.max()) + 10, last + 150))
    s, e = _window(min(int(r0.armed_idx), int(r0.trig)) - 25, hi_end, len(dm))
    path = f"{outdir}/{tag}_M15.png"
    plot_structure(dm, P["M15"]["sw"], P["M15"]["ev"], P["M15"]["fv"], s, e, path,
                   f"M15 {r0.direction} | {len(g)} trade(s) | zone: {label}",
                   obs=P["M15"]["ob"], fibs=_fib(r0, to_m, int(r0.armed_idx) + 3),
                   highlights=[dict(x0=int(np.searchsorted(tm, t1[_zone_x0(r0)])), x1=int(r0.armed_idx) + 1,
                                    bottom=r0.zone_bottom, top=r0.zone_top, color="#ef6c00", label=label)],
                   positions=_positions(g, len(dm)),
                   vlines=[(int(r0.armed_idx), "zone touched", "#1976d2")])
    paths.append(path)
    return paths