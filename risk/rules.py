"""
Trade-execution rules shared by every strategy: how a POI becomes an entry, where the stop goes,
and how a trade is played forward. Pip-based numbers live in ExecutionRules (gold's values are the
defaults); the pip size itself comes from the pair (config/pairs.yaml -> PairConfig.pip_size).
"""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ExecutionRules:
    big_poi_pips: float = 40    # a POI taller than this is entered half-way, not at its edge
    max_poi_pips: float = 60    # a POI taller than this is not traded
    sl_step_pips: float = 5     # stop distance is rounded UP to a multiple of this


DEFAULT_RULES = ExecutionRules()


def poi_entry(bottom, top, bull, pip, rules=DEFAULT_RULES):
    """
    Entry rule for a zone POI. Returns (height_pips, entry, far, mode).
      height_pips  zone height in pips
      entry        near edge (mode "edge"), or the middle of the zone when it is taller than
                   rules.big_poi_pips (mode "half")
      far          the zone's far edge (bottom for a bull POI, top for a bear POI)
    entry, far and mode are None when the zone is taller than rules.max_poi_pips (not traded).
    """
    height = (top - bottom) / pip
    if height > rules.max_poi_pips:
        return height, None, None, None
    big = height > rules.big_poi_pips
    entry = (top + bottom) / 2 if big else (top if bull else bottom)
    far = bottom if bull else top
    return height, entry, far, ("half" if big else "edge")


def stop_loss(entry, far, bull, pip, rules=DEFAULT_RULES):
    """Stop covering the whole zone, with its distance rounded up to a multiple of
    rules.sl_step_pips. Returns (sl_price, distance_in_price)."""
    step = rules.sl_step_pips * pip
    dist = np.ceil(abs(entry - far) / step - 1e-9) * step
    return (entry - dist if bull else entry + dist), dist


def simulate(low, high, i, bull, sl, tp):
    """
    Plays a trade forward from the entry candle i. Returns (exit_idx, "SL" | "TP" | None).
    Stop is checked first (so it wins when both are hit in one candle); a take-profit only counts
    from the candle after entry. A NaN tp never triggers. None = still open at the last candle.
    """
    n = len(low)
    for j in range(i, n):
        if (low[j] <= sl) if bull else (high[j] >= sl):
            return j, "SL"
        if j > i and tp == tp and ((high[j] >= tp) if bull else (low[j] <= tp)):
            return j, "TP"
    return n - 1, None