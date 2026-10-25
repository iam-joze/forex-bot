from dataclasses import dataclass
from pathlib import Path
import yaml

ROLES = ("bias", "trend", "structure", "entry")
CONFIG_PATH = Path(__file__).with_name("pairs.yaml")


@dataclass(frozen=True)
class PairConfig:
    name: str
    broker_symbol: str
    volatility_class: str
    pip_size: float
    timeframes: dict  # role -> timeframe string

    def tf(self, role: str) -> str:
        return self.timeframes[role]


@dataclass(frozen=True)
class Settings:
    broker: str
    symbol_suffix: str
    pairs: dict  # name -> PairConfig


def load_settings(path=CONFIG_PATH) -> Settings:
    raw = yaml.safe_load(Path(path).read_text())
    suffix = raw["broker"].get("symbol_suffix", "")
    role_maps = raw["timeframe_roles"]

    for cls, mapping in role_maps.items():
        missing = set(ROLES) - set(mapping)
        if missing:
            raise ValueError(f"Class '{cls}' missing roles: {missing}")

    pairs = {}
    for name, p in raw["pairs"].items():
        cls = p["volatility_class"]
        if cls not in role_maps:
            raise ValueError(f"{name}: unknown volatility_class '{cls}'")
        pairs[name] = PairConfig(
            name=name,
            broker_symbol=name + suffix,
            volatility_class=cls,
            pip_size=float(p["pip_size"]),
            timeframes=dict(role_maps[cls]),
        )
    return Settings(raw["broker"]["name"], suffix, pairs)