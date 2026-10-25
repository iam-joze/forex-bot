from abc import ABC, abstractmethod
from datetime import datetime
import pandas as pd

# Candle DataFrame contract: columns time (candle OPEN time), open, high, low, close, volume


class Broker(ABC):
    @abstractmethod
    def connect(self) -> None: ...

    @abstractmethod
    def disconnect(self) -> None: ...

    def prepare_symbol(self, symbol: str) -> None:
        """Make a symbol available for data requests. No-op by default."""

    @abstractmethod
    def get_candles_range(self, symbol: str, timeframe: str,
                          start: datetime, end: datetime) -> pd.DataFrame: ...

    @abstractmethod
    def get_candles(self, symbol: str, timeframe: str, count: int,
                    include_forming: bool = False) -> pd.DataFrame: ...