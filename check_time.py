import MetaTrader5 as mt5
from datetime import datetime, timezone

SYMBOL = "EURUSDm"
mt5.initialize()
mt5.symbol_select(SYMBOL, True)

tick = mt5.symbol_info_tick(SYMBOL)
utc_now = datetime.now(timezone.utc)
print("VM local time :", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
print("UTC now       :", utc_now.strftime("%Y-%m-%d %H:%M:%S"))
print("Last tick time:", datetime.fromtimestamp(tick.time, timezone.utc).strftime("%Y-%m-%d %H:%M:%S"), "(as the broker stamps it)")
offset = (tick.time - utc_now.timestamp()) / 3600
print(f"Broker server offset vs UTC ≈ {offset:+.2f} h  (round to nearest 0.5)")

for name, tf in [("H1", mt5.TIMEFRAME_H1), ("D1", mt5.TIMEFRAME_D1)]:
    r = mt5.copy_rates_from_pos(SYMBOL, tf, 0, 3)
    print(name, "candle open times:", [datetime.fromtimestamp(x["time"], timezone.utc).strftime("%m-%d %H:%M") for x in r])
mt5.shutdown()