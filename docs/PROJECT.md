# Forex Bot: Project Reference

Purpose: a trading robot built in Python on MetaTrader 5 data. The project is a set of small, separately
reviewable components (data, structure detection, strategies, risk rules), each with its own review
script that prints statistics and saves charts.

How to use this file: paste or attach it at the start of a new chat. It describes every file and the data
contracts between them, so a change can usually be requested without sharing the code. If a change touches
something not described here, ask for that one file or function only.

Last updated: 2026-11-17 (repo commit `ec05170`). The project is open to change and extension, so treat
this file as a snapshot and update it when the code changes.

---

## 1. Rules for the AI

1. Work on one thing at a time, even when several are suggested at once.
2. For anything new, ask all the necessary questions before building anything.
3. Always give code together with its precise location: the file, the function, and either the line range
   or the exact existing text it replaces. When several edits are given for one file, state the order to
   apply them in (bottom up keeps line numbers valid). A new file is given in full with its path.
4. When a component that has its own review script is changed, test it with that script and look at charts
   that reflect only that component (section 8 lists which script covers which component).
5. The confluence review (`reviews/strategy/run_confluence_review.py`) runs only when the confluence
   engine, or something it depends on directly (`risk/rules.py`), changes.
6. Imports are absolute from the project root, for example `from strategy.confluence import scan` and
   `from risk.rules import DEFAULT_RULES`.
7. Review scripts are run as modules from the project root, for example
   `py -m reviews.structure.run_poi_review`. Every package folder has an empty `__init__.py`, except
   `logs/`, which only holds output.
8. Nothing may use information from the future: an indicator value on candle `i` may only use candles up to
   and including `i`, and a higher-timeframe candle counts only once it has closed.

## 2. Environment

| Item | Value |
|---|---|
| OS and shell | Windows, PowerShell, project at `C:\dev\forex-bot`, run from the root with the `.venv` active |
| Python | 3.11 |
| Packages | `requirements.txt` (pandas, numpy, matplotlib, pyyaml, MetaTrader5, smartmoneyconcepts) |
| Broker | MetaTrader 5 through the `MetaTrader5` package, Exness cent account (symbols carry the suffix `m`, for example `XAUUSDm`) |
| Output | Charts and review output go to `logs/<review name>/` (git-ignored) |

Candle table contract (every broker and loader function returns this): columns `time` (candle OPEN time,
naive timestamp, treated as UTC), `open`, `high`, `low`, `close`, `volume`.

## 3. Repository map

### Root

| File | What it does |
|---|---|
| `test_mt5.py` | Smoke test: connects to MT5, prints the account and the last 5 candles of H1 and M15 for one symbol |
| `run_data_demo.py` | Demo of the role-based loader: `load_mtf` and `align_mtf` for two pairs, with a no-lookahead assertion |
| `run_smc_compare.py` | Compares this project's swings, BOS/MSS, FVG and OB against the `smartmoneyconcepts` library and saves side-by-side charts to `logs/smc/` |
| `requirements.txt` | Python dependencies |
| `.gitignore` | Ignores `.venv/`, `.env`, `logs/`, `__pycache__/` |

### `broker/`

| File | What it does |
|---|---|
| `base.py` | `Broker` abstract class: `connect()`, `disconnect()`, `prepare_symbol(symbol)`, `get_candles_range(symbol, timeframe, start, end)`, `get_candles(symbol, timeframe, count, include_forming=False)` |
| `mt5.py` | `MT5Broker(Broker)`: the MT5 implementation. Timeframes `M1, M5, M15, M30, H1, H4, D1`. `tick_volume` becomes `volume` |

### `config/`

| File | What it does |
|---|---|
| `pairs.yaml` | Broker name and symbol suffix, `timeframe_roles` per volatility class (`low`, `high`), and the pair list. Each pair has `volatility_class` and `pip_size` |
| `settings.py` | `load_settings()` returns `Settings(broker, symbol_suffix, pairs)`. `PairConfig` has `name`, `broker_symbol` (name plus suffix), `volatility_class`, `pip_size`, `timeframes` and `.tf(role)`. `ROLES = ("bias", "trend", "structure", "entry")` |

Pairs currently listed: EURUSD, USDJPY, GBPUSD, XAUUSD, USDCAD, USDCHF, AUDUSD. XAUUSD uses `pip_size: 0.1`.
BTCUSD is planned and is not in the file yet (its `pip_size` has to be defined when it is added).

### `data/`

| File | What it does |
|---|---|
| `timeframes.py` | `TF_MINUTES` and `tf_delta(tf)` (a pandas Timedelta for a timeframe string) |
| `resample.py` | `resample_ny_anchor(h1, tf)`: builds H4 and D1 candles from H1, anchored to 17:00 New York time (matches TradingView) |
| `loader.py` | `fetch_tf(broker, symbol, tf, start, end)`: the single fetching primitive (H4 and D1 are built from H1, other timeframes come straight from the broker). `load_mtf(broker, pair, start, end, warmup_bars=100)` fetches every role's timeframe. `align_mtf(frames, pair, start, base_role="entry")` gives one row per entry candle with the latest already-closed candle of each other role (columns `{role}_parent_time`, `{role}_closed_open/high/low/close`, `{role}_closed_at`) |

Two ways of choosing timeframes exist. The role map (`load_mtf`, `align_mtf`, `timeframe_roles` in the yaml)
was written when a single strategy was expected, and only the two demo scripts use it. Everything else,
including the confluence engine, picks its own timeframes and history length and calls `fetch_tf`. New
strategies should declare their own timeframes.

### `structure/`

All detectors take a candle table and return a table. Indices (`idx`) are row positions in that table.

| File | What it does |
|---|---|
| `utils.py` | `atr(df, period=14)`: simple-average ATR, no lookahead |
| `myswings.py` | `find_swings(df, n=2)`: fractal swings. `alternate(swings)`: forces strict high/low alternation. `find_swings_zigzag(df, atr_mult=2.0, atr_period=14)`: ATR zigzag. Output columns `SWING_COLS = idx, time, price, kind ("high" or "low"), confirmed_idx`. A swing is only known from `confirmed_idx` |
| `swing_methods.py` | Chooses the swing detector per timeframe. `SWING_METHOD = {"H1": ("fractal", {"n": 5})}`, `DEFAULT_METHOD = ("fractal", {"n": 5})`, and `swings_for(df, tf)` returns the swings for that timeframe. Timeframes without an entry use the default. This is where a new swing method gets registered |
| `choch_mss.py` | `find_structure(df, swings, atr_period=14)`: BOS (break in the trend direction, a wick is enough) and MSS (break against the trend, needs a close). Columns `EVENT_COLS = idx, time, direction, type, broken_idx, level, origin_idx, leg_atr` |
| `fvg.py` | `find_fvgs(df, min_atr_mult=0.3, atr_period=14)`: raw 3-candle imbalances (columns `c1, c2, c3, time, direction, bottom, top, size`). `validate_fvgs(fvgs, events, df)`: keeps FVGs that start the impulse leg of a BOS/MSS; adds `event_type, break_idx, break_time, leg_order, known_idx, known_time, mitigated_idx, mitigated_time, status` (`active` or `mitigated`) |
| `order_blocks.py` | `find_order_blocks(df, events, min_leg_atr=2.0, max_after_origin=5, bear_close_ref="low", liq_lookback=40, liq_swing_n=2)`: one order block per strong BOS/MSS leg. Columns `OB_COLS = idx, time, direction, high, low, body_high, body_low, event_type, break_idx, break_time, leg_atr, fvg_bottom, fvg_top, known_idx, known_time, mitigated_idx, mitigated_time, status, liq_grab, liq_level`. `df.attrs["funnel"]` counts legs surviving each rule. It calls `find_swings(df, 2)` directly for the liquidity-grab note, on purpose |
| `fib_zones.py` | Fib of a BOS/MSS leg and the judging of order blocks and FVGs against it. `fib_state(df, events, swings, ote=(0.62, 0.79), carry_over=False)`: fib anchored only to confirmed swings. `evaluate_pois(df, fib, obs, fvgs)`: side of 0.5 and OTE overlap for each POI. `provisional_fib(df, events, swings, double_top_tol_atr=0.3, atr_period=14)`: fib live from the break candle (columns `dir, start, end, start_idx, end_idx, leg_event_idx, leg_valid_to, mid, lvl62, lvl79, ote_lo, ote_hi`). `validate_pois(df, fib, obs, fvgs, expire_on_next_event=True)`: a POI becomes valid when its whole zone sits on the right side of 0.5, then records the first entry touch (columns `POI_COLS`: `kind, idx, direction, bottom, top, event_idx, known_idx, status ("touched", "untouched", "never_valid"), valid_from, valid_to, delay, early_touch, touch_idx, entry, sl_ref, ote, snap_idx, fib_*, mid, lvl62, lvl79, ote_lo, ote_hi`) |

### `risk/`

| File | What it does |
|---|---|
| `rules.py` | Trade-execution rules shared by all strategies. `ExecutionRules(big_poi_pips=40, max_poi_pips=60, sl_step_pips=5)` and `DEFAULT_RULES`. `poi_entry(bottom, top, bull, pip, rules)` returns `(height_pips, entry, far, mode)` (entry at the near edge, or half-way for tall zones, or `None` when the zone is too tall). `stop_loss(entry, far, bull, pip, rules)` returns `(sl_price, distance)` with the distance rounded up to a multiple of `sl_step_pips`. `simulate(low, high, i, bull, sl, tp)` plays a trade forward and returns `(exit_idx, "SL" / "TP" / None)` (stop first when both are hit in one candle, take-profit only from the candle after entry). The pip size is never stored here: it comes from `PairConfig.pip_size` |

### `strategy/`

| File | What it does |
|---|---|
| `confluence.py` | The first strategy engine (see section 6). `prepare(df, tf, full=False, fvg_min_atr=0.3, min_leg_atr=2.0)` builds the structure tables for one timeframe. `scan(P, pair, rules=DEFAULT_RULES, log=None)` returns the trade table. Strategy-specific constants at the top: `MOM_ATR`, `ZONE_FAIL_KILLS`, `EXCLUDE_MITIGATED`, `OTE`, `DUR` |

### `tools/`

| File | What it does |
|---|---|
| `plot.py` | `plot_structure(df, swings, events, fvgs, start, end, path, title="", obs=None, fibs=None, highlights=None, positions=None, vlines=None)`: draws candles `[start:end]` with swings, BOS/MSS lines, FVG boxes, order blocks, fibs, outlined boxes, long/short position boxes and vertical lines, then saves a PNG. Every review uses it |
| `plot_signal.py` | `plot_entry_set(P, g, outdir, tag)`: for the trades `g` of one leg, saves one chart per timeframe of the confluence engine (fib and touched zone on the higher timeframe, MSS and positions on the lower one). It depends on the confluence trade-table columns |

### `calibration/`

`swings_H1.csv` and `swings_M15.csv`: swings marked by hand, columns `time_utc,kind` (`high` or `low`). They are
the ground truth for `run_swing_eval.py` and must be marked on the pair set in that script.

### `reviews/`

Review and diagnostic scripts: `structure/` holds one review per structure component, `strategy/` holds the
strategy-engine review and its trace tool, and `check_time.py` is a broker diagnostic. Section 8 lists each
script with its command.

### `backtest/`, `notify/`, `live/`, `tests/`

Empty packages (only `__init__.py`) reserved for later (see section 9).

## 4. Data contracts (tables passed between components)

| Table | Produced by | Key columns |
|---|---|---|
| Candles | broker, `fetch_tf` | `time, open, high, low, close, volume` |
| Swings | `find_swings`, `alternate`, `swings_for` | `idx, time, price, kind, confirmed_idx` |
| Events | `find_structure` | `idx, time, direction ("bull"/"bear"), type ("BOS"/"MSS"), broken_idx, level, origin_idx, leg_atr` |
| FVGs | `validate_fvgs` | `bottom, top, direction, c1, c2, c3, known_idx, mitigated_idx, status, break_idx, ...` |
| Order blocks | `find_order_blocks` | `idx, high, low, direction, known_idx, mitigated_idx, status, break_idx, leg_atr, liq_grab, ...` |
| Fib | `fib_state`, `provisional_fib`, `swing_leg_fib` in `confluence.py` | one row per candle: `dir, start, end, start_idx, end_idx, mid, lvl62, lvl79, ote_lo, ote_hi` plus leg ids |
| POIs | `validate_pois`, `pois_on_fib` in `confluence.py` | `kind ("OB"/"FVG"), idx, direction, bottom, top, valid_from, valid_to, ...` |
| Trades | `scan` | one row per trade (below); `attrs["stats"]` holds the funnel counts |

Trade table columns: `time, idx, direction, entry, sl, tp, sl_pips, tp_pips, rr, kind, entry_mode ("edge"/"half"),
ote, m15_leg, seq, zone_kind, zone_idx, fib_leg, fib_start, fib_end, fib_mid, fib_lo, fib_hi, fib_start_idx,
fib_end_idx, poi_idx, armed_idx, armed_k1, trig, zone_bottom, zone_top, approach_rng_atr, touch_rng_atr,
exit_idx, outcome ("SL"/"TP"/None)`.

## 5. Swing detection

- `structure/swing_methods.py` is the single switch: `swings_for(df, tf)` returns the swings for a
  timeframe, using `SWING_METHOD` or `DEFAULT_METHOD`. Only `strategy/confluence.py` calls it.
- The structure review scripts (`run_structure_demo`, `run_ob_review`, `run_fib_review`, `run_poi_review`)
  and `run_smc_compare.py` currently call `alternate(find_swings(df, SWING_N))` directly with their own
  `SWING_N = 5`. They will be switched to `swings_for` when the new swing method is added.
- A new swing method for M15 is pending. Until it exists, M15 uses the default (fractal n=5 with alternation).
- Candidate detectors are scored against the hand-marked calibration files with `run_swing_eval.py`.

## 6. Strategy engines

Several strategy engines are expected, each with its own timeframes and rules. They live in `strategy/` as
separate files. The existing `confluence.py` is one example and may be rewritten or deleted.

An engine is expected to follow this shape (not enforced by code yet):
- `prepare(df, tf, ...)` builds the structure tables for one timeframe from a candle table and returns a dict.
- `scan(P, pair, rules=DEFAULT_RULES, log=None)` takes those dicts, the `PairConfig` (pip size) and the
  execution rules, and returns a trade table. It uses `risk/rules.py` for entry, stop and trade simulation,
  and appends `(time, message)` notes to `log` when one is given.
- Higher-timeframe candles count only after they close (`close_times`), so there is no lookahead.

How the existing `confluence.py` works: swings per timeframe, then BOS/MSS, FVGs and order blocks. On the
higher timeframe it takes the fib of the current swing leg (the last two confirmed alternating swings) and
finds the zone POIs on the right side of 0.5. On the lower timeframe it follows the leg until price touches
an active zone (armed), waits for an MSS in the leg direction (trigger), then enters on the first touch of a
lower-timeframe POI. Stop and entry come from `risk/rules.py`, the take-profit is the extreme of the
lower-timeframe leg, one position is open at a time, and a leg is dead when price closes beyond its end or
its start.

## 7. Where to change what

| Goal | Where |
|---|---|
| Use another swing detector for a timeframe | `SWING_METHOD` in `structure/swing_methods.py` |
| Add a new swing algorithm | New function in `structure/` (returning `SWING_COLS`), then a branch in `swings_for` |
| Change entry, stop or trade-simulation rules, or the pip thresholds | `risk/rules.py` |
| Add or change a pair, pip size, symbol suffix | `config/pairs.yaml` |
| Add a strategy | New file in `strategy/` following section 6, and a review script in `reviews/strategy/` |
| Change how charts look | `tools/plot.py` (structure items), `tools/plot_signal.py` (strategy entries) |
| Fetch data | `data/loader.py` (`fetch_tf`) |

## 8. Review scripts

Run from the project root with `py -m <module>`. The pair for a review is the `PAIR` constant at the top
of each script. Reviews are meant to be made on XAUUSD, and BTCUSD is planned.

| Component changed | Script and command | Output | `PAIR` now |
|---|---|---|---|
| Swings | `reviews.structure.run_swing_eval`: scores fractal and zigzag rules against `calibration/` | printed table only (no charts yet) | EURUSD |
| Swings, BOS/MSS, FVG | `reviews.structure.run_structure_demo`: charts for each role timeframe | `logs/structure/` | EURUSD |
| Order blocks | `reviews.structure.run_ob_review` | stats and `logs/obs/` | EURUSD |
| Fib on confirmed swings, POI side of 0.5 | `reviews.structure.run_fib_review` | stats and `logs/fib/` | EURUSD |
| Provisional fib, POI validity and entry touch | `reviews.structure.run_poi_review` | stats and `logs/poi/` | XAUUSD |
| Confluence engine, `risk/rules.py` | `reviews.strategy.run_confluence_review`: funnel, trade table, self-check, no-lookahead cut test | `logs/confluence/` | XAUUSD |
| Explain the confluence engine around a time | `reviews.strategy.trace_setup` (optionally with times, for example `"2026-03-01 14:30"`) | printed notes | XAUUSD |
| Broker time offset and latest candle times | `reviews.check_time` (checks delay in data retrieval) | printed | symbol `EURUSDm` |
| Own detectors versus the SMC library | `py run_smc_compare.py` | `logs/smc/` | XAUUSD |
| Data loader | `py run_data_demo.py` | printed | EURUSD, GBPUSD |

## 9. Roadmap (open to change and extension)

- Define a new swing method (for M15 first), with its own review that prints statistics and saves charts.
  `run_swing_eval.py` currently prints only, so it gets charts as part of this.
- Switch the structure reviews and `run_smc_compare.py` from `find_swings` to `swings_for`.
- Write several strategy engines, each for a different strategy and with its own timeframes. The existing
  `confluence.py` may be deleted. Pairs for each strategy will be chosen later (XAUUSD now, BTCUSD planned).
- Optional per-pair overrides of `ExecutionRules` in `config/pairs.yaml`.
- Decide whether the role map in `config/pairs.yaml` stays, once a second strategy exists.
- Empty folders and their intended roles:
  - `data/`: multi-timeframe fetch and alignment, per-pair and config driven (partly present)
  - `backtest/`: backtesting engine
  - `notify/`: Telegram notifications (`telegram.py`)
  - `risk/`: risk management with SL and TP (`risk_manager.py`); `rules.py` already exists
  - `live/`: paper trading and live trading loop
  - `tests/`: automated tests of the project code
