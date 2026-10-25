from datetime import datetime
import MetaTrader5 as mt5
import pandas as pd
from .base import Broker

_TF = {
    "M1": mt5.TIMEFRAME_M1, "M5": mt5.TIMEFRAME_M5, "M15": mt5.TIMEFRAME_M15,
    "M30": mt5.TIMEFRAME_M30, "H1": mt5.TIMEFRAME_H1, "H4": mt5.TIMEFRAME_H4,
    "D1": mt5.TIMEFRAME_D1,
}


class MT5Broker(Broker):
    def connect(self) -> None:
        if not mt5.initialize():
            raise ConnectionError(f"MT5 initialize failed: {mt5.last_error()}")

    def disconnect(self) -> None:
        mt5.shutdown()

    def prepare_symbol(self, symbol: str) -> None:
        if not mt5.symbol_select(symbol, True):
            raise ValueError(f"Cannot select symbol {symbol}: {mt5.last_error()}")

    @staticmethod
    def _to_df(rates, symbol: str, timeframe: str) -> pd.DataFrame:
        if rates is None or len(rates) == 0:
            raise RuntimeError(f"No data for {symbol} {timeframe}: {mt5.last_error()}")
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df = df.rename(columns={"tick_volume": "volume"})
        return df[["time", "open", "high", "low", "close", "volume"]].reset_index(drop=True)

    def get_candles_range(self, symbol, timeframe, start: datetime, end: datetime):
        self.prepare_symbol(symbol)
        rates = mt5.copy_rates_range(symbol, _TF[timeframe], start, end)
        return self._to_df(rates, symbol, timeframe)

    def get_candles(self, symbol, timeframe, count: int, include_forming: bool = False):
        self.prepare_symbol(symbol)
        pos = 0 if include_forming else 1  # position 0 is the still-forming candle
        rates = mt5.copy_rates_from_pos(symbol, _TF[timeframe], pos, count)
        return self._to_df(rates, symbol, timeframe)