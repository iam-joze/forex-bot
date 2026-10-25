import MetaTrader5 as mt5
import pandas as pd

SYMBOL = "EURUSDm"

if not mt5.initialize():
    print("init failed:", mt5.last_error())
    quit()

print("Account:", mt5.account_info().login, "| Server:", mt5.account_info().server)

# Make sure the symbol is visible in Market Watch
if not mt5.symbol_select(SYMBOL, True):
    print("symbol_select failed:", mt5.last_error())

for name, tf in [("H1", mt5.TIMEFRAME_H1), ("M15", mt5.TIMEFRAME_M15)]:
    rates = mt5.copy_rates_from_pos(SYMBOL, tf, 0, 5)
    if rates is None:
        print(name, "no data:", mt5.last_error())
        continue
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    print(f"\n{name}\n", df[["time", "open", "high", "low", "close"]])

mt5.shutdown()