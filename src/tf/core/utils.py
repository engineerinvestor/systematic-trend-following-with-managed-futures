from dataclasses import dataclass
from typing import Dict, List
import numpy as np
import pandas as pd

def annualize_vol(daily_vol: float, periods: int = 252) -> float:
    return daily_vol * np.sqrt(periods)

def deannualize_vol(ann_vol: float, periods: int = 252) -> float:
    return ann_vol / np.sqrt(periods)

def rolling_vol(returns: pd.Series, window: int = 63) -> pd.Series:
    # The warm-up stays NaN: the old bfill filled it from a future estimate
    # (lookahead), and fillna(method=) was removed in pandas 3 anyway.
    return returns.rolling(window).std()

@dataclass
class ContractMeta:
    symbol: str
    sector: str
    point_value: float
