import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def plot_structure(df, swings, events, fvgs, start, end, path, title="", ranges=None):
    """Candles [start:end] with swings, BOS/MSS lines and validated FVG boxes."""
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
            ax.text(e.idx, e.level, e.type, color=col, fontsize=8, va="bottom", zorder=5)

    if len(fvgs):
        for f in fvgs.itertuples(index=False):
            x1 = f.mitigated_idx if f.mitigated_idx == f.mitigated_idx and f.mitigated_idx is not None else end
            if x1 < start or f.c1 >= end:
                continue
            col = "#2e7d32" if f.direction == "bull" else "#c62828"
            ax.add_patch(Rectangle((max(f.c1, start), f.bottom), min(x1, end) - max(f.c1, start),
                                   f.top - f.bottom, color=col, alpha=0.22 if f.status == "active" else 0.12,
                                   zorder=1))

    if ranges is not None and len(ranges):
        for r in ranges.itertuples(index=False):
            x0, x1 = max(r.start_idx, start), min(r.end_idx + 1, end)
            if x1 <= start or r.start_idx >= end:
                continue
            ax.add_patch(Rectangle((x0, r.bottom), x1 - x0, r.top - r.bottom, facecolor="#1565c0",
                                   edgecolor="#1565c0", alpha=0.10, lw=1.2, zorder=0))
            ax.text(x0 + 0.5, r.top, "RANGE", color="#1565c0", fontsize=8, va="bottom", zorder=5)
            if r.status == "broken" and start <= r.break_idx < end:
                up = r.break_dir == "up"
                strong = r.momentum >= 1.0
                ax.annotate(f"{r.momentum:.1f}", (r.break_idx, r.top if up else r.bottom),
                            xytext=(0, 14 if up else -14), textcoords="offset points",
                            ha="center", va="center", fontsize=9, fontweight="bold" if strong else "normal",
                            color="#1565c0", zorder=6,
                            arrowprops=dict(arrowstyle="-|>", color="#1565c0", lw=1.4 if strong else 0.8))

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
    fig.savefig(path, dpi=110)
    plt.close(fig)