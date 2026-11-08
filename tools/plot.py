import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def plot_structure(df, swings, events, fvgs, start, end, path, title="", obs=None, fibs=None):
    """Candles [start:end] with swings, BOS/MSS lines, validated FVG boxes, order blocks and fibs."""
    end = min(end, len(df))
    fig, ax = plt.subplots(figsize=(16, 8))

    for i in range(start, end):
        r = df.iloc[i]
        up = r["close"] >= r["open"]
        col = "#26a69a" if up else "#ef5350"
        ax.plot([i, i], [r["low"], r["high"]], color=col, lw=0.8, zorder=2)
        body_lo, body_hi = min(r["open"], r["close"]), max(r["open"], r["close"])
        ax.add_patch(Rectangle((i - 0.3, body_lo), 0.6, max(body_hi - body_lo, 1e-9),
                               color=col, zorder=3))

    for s in swings.itertuples(index=False):
        if start <= s.idx < end:
            ax.scatter(s.idx, s.price, marker="v" if s.kind == "high" else "^",
                       s=25, color="black", zorder=4)

    for e in events.itertuples(index=False):
        if start <= e.idx < end or start <= e.broken_idx < end:
            x0 = max(e.broken_idx, start)
            col = "#1565c0" if e.type == "BOS" else "#6a1b9a"
            ax.plot([x0, e.idx], [e.level, e.level], color=col, lw=1.2, ls="--", zorder=4)
            ax.text(e.idx, e.level, e.type, color=col, fontsize=8, va="bottom", zorder=5,
                    clip_on=True)

    if fvgs is not None and len(fvgs):
        for f in fvgs.itertuples(index=False):
            x1 = f.mitigated_idx if f.mitigated_idx == f.mitigated_idx and f.mitigated_idx is not None else end
            if x1 < start or f.c1 >= end:
                continue
            col = "#2e7d32" if f.direction == "bull" else "#c62828"
            ax.add_patch(Rectangle((max(f.c1, start), f.bottom), min(x1, end) - max(f.c1, start),
                                   f.top - f.bottom, color=col, alpha=0.22 if f.status == "active" else 0.12,
                                   zorder=1))

    if obs is not None and len(obs):
        for o in obs.itertuples(index=False):
            x1 = o.mitigated_idx if o.mitigated_idx == o.mitigated_idx and o.mitigated_idx is not None else end
            if x1 < start or o.idx >= end:
                continue
            col = "#00838f" if o.direction == "bull" else "#ef6c00"
            x0 = max(o.idx, start)
            ax.add_patch(Rectangle((x0 - 0.5, o.low), min(x1, end) - x0 + 0.5, o.high - o.low,
                                   facecolor=col, edgecolor=col, alpha=0.20, lw=1.6, zorder=1))
            ax.text(x0, o.high if o.direction == "bear" else o.low,
                    "OB" + (" G" if o.liq_grab else ""), color=col, fontsize=9, fontweight="bold",
                    va="bottom" if o.direction == "bear" else "top", zorder=6, clip_on=True)

    if fibs:
        for fb in fibs:
            xs = max(fb["x_start"], start)
            xe = min(fb["x_to"], end)
            ax.plot([fb["x_start"], fb["x_end"]], [fb["start"], fb["end"]], color="#455a64", lw=1.4,
                    marker="o", ms=4, zorder=4)
            ax.add_patch(Rectangle((xs, fb["ote_lo"]), xe - xs, fb["ote_hi"] - fb["ote_lo"],
                                   facecolor="#fbc02d", alpha=0.18, edgecolor="none", zorder=0))
            for price, label, col, ls in ((fb["mid"], "0.5", "#455a64", "--"),
                                          (fb["lvl62"], "0.62", "#f9a825", "-"),
                                          (fb["lvl79"], "0.79", "#f9a825", "-")):
                ax.plot([xs, xe], [price, price], color=col, lw=1.0, ls=ls, zorder=4)
                ax.text(xe, price, " " + label, color=col, fontsize=8, ha="left", va="center",
                        clip_on=True, zorder=6)

    lo = df["low"].iloc[start:end].min()
    hi = df["high"].iloc[start:end].max()
    pad = (hi - lo) * 0.03
    ax.set_xlim(start - 1, end + 1)
    ax.set_ylim(lo - pad, hi + pad)
    step = max(1, (end - start) // 10)
    ticks = list(range(start, end, step))
    ax.set_xticks(ticks)
    ax.set_xticklabels([df["time"].iloc[t].strftime("%m-%d %H:%M") for t in ticks], rotation=30, fontsize=8)
    ax.set_title(title)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)