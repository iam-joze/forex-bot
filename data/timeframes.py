import pandas as pd

TF_MINUTES = {"M1": 1, "M5": 5, "M15": 15, "M30": 30, "H1": 60, "H4": 240, "D1": 1440}


def tf_delta(tf: str) -> pd.Timedelta:
    return pd.Timedelta(minutes=TF_MINUTES[tf])