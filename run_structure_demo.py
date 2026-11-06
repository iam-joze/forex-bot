from datetime import datetime, timedelta, timezone
from broker.mt5 import MT5Broker
from config.settings import load_settings
from data.loader import load_mtf
from structure.swings import find_swings, alternate 
from structure.choch_mss import find_structure
from structure.fvg import find_fvgs, validate_fvgs
from tools.plot import plot_structure

PAIR = "EURUSD"
DAYS = 120
SWING_N = 5
FVG_MIN_ATR = 0.3
SHOW = 150  # candles per chart

settings = load_settings()
pair = settings.pairs[PAIR]

broker = MT5Broker()
broker.connect()
try:
    end = datetime.now(timezone.utc).replace(tzinfo=None)
    frames = load_mtf(broker, pair, end - timedelta(days=DAYS), end)
finally:
    broker.disconnect()

for role, df in frames.items():
    df = df.reset_index(drop=True)
    tf = pair.tf(role)
    sw = alternate(find_swings(df, SWING_N))
    ev = find_structure(df, sw)
    fv = validate_fvgs(find_fvgs(df, FVG_MIN_ATR), ev, df)

    print(f"{role:9} {tf:4} candles={len(df):5} swings={len(sw):4} "
          f"BOS={int((ev.type == 'BOS').sum()):3} MSS={int((ev.type == 'MSS').sum()):3} "
          f"valid FVGs={len(fv):3} active={int((fv.status == 'active').sum()) if len(fv) else 0}")
    path = f"logs/structure/{PAIR}_{role}_{tf}.png"
    plot_structure(df, sw, ev, fv, max(0, len(df) - SHOW), len(df), path,
                   f"{PAIR} {tf} ({role}) swing_n={SWING_N} fvg_min_atr={FVG_MIN_ATR}")
    print("   chart:", path)