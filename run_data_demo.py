from datetime import datetime, timedelta, timezone
import pandas as pd
from broker.mt5 import MT5Broker
from config.settings import load_settings, ROLES
from data.loader import load_mtf, align_mtf

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 30)

settings = load_settings()
broker = MT5Broker()
broker.connect()

try:
    for name in ("EURUSD", "GBPUSD"):  # one low-vol, one high-vol
        pair = settings.pairs[name]
        end = datetime.now(timezone.utc).replace(tzinfo=None)
        start = end - timedelta(days=30)

        frames = load_mtf(broker, pair, start, end)
        df = align_mtf(frames, pair, start)

        print(f"\n=== {name} ({pair.volatility_class}) roles: {pair.timeframes} ===")
        print(f"rows: {len(df)}")
        cols = ["time", "close", "bias_parent_time", "bias_closed_close",
                "trend_closed_close", "structure_closed_close"]
        print(df[cols].tail(5))

        # no-lookahead check
        for r in ROLES:
            if r != "entry":
                assert (df[f"{r}_closed_at"] <= df["close_time"]).all(), f"lookahead in {r}"
        print("lookahead check: OK")
finally:
    broker.disconnect()