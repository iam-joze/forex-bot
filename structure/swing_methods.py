"""Which swing detector each timeframe uses. Edit SWING_METHOD to switch a timeframe."""
from structure.myswings import find_swings, alternate

SWING_METHOD = {
    "H1": ("fractal", {"n": 5}),             # structure/myswings.py: fractal n=5 + alternation
}
DEFAULT_METHOD = ("fractal", {"n": 5})


def swings_for(df, tf):
    method, params = SWING_METHOD.get(tf, DEFAULT_METHOD)
    if method == "fractal":
        return alternate(find_swings(df, params.get("n", 5)))
    raise ValueError(f"unknown swing method {method!r} for timeframe {tf}")