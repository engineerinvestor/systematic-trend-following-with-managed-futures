"""Volatility estimators used for sizing and signal normalization."""

from __future__ import annotations

import numpy as np
import pandas as pd


def com_to_lambda(center_of_mass: float) -> float:
    """Convert an EWMA center of mass to the equivalent decay factor.

    Trend-following research states EWMA volatility windows as a center of mass
    (Moskowitz, Ooi, and Pedersen use 60 days; CTA replication work commonly uses
    40), while :func:`ewma_vol` takes a RiskMetrics decay factor. The two are
    related by ``com = lam / (1 - lam)``.
    """

    if center_of_mass <= 0:
        raise ValueError("center_of_mass must be positive")
    return float(center_of_mass) / (float(center_of_mass) + 1.0)


def ewma_vol(
    returns: pd.DataFrame,
    lam: float = 0.94,
    min_periods: int = 20,
    *,
    annualize: bool = True,
    periods_per_year: int = 252,
    center_of_mass: float | None = None,
) -> pd.DataFrame:
    """Exponentially weighted volatility estimate.

    Pass either ``lam`` (a RiskMetrics decay factor) or ``center_of_mass``; the
    latter takes precedence and is converted with :func:`com_to_lambda`.
    """

    if center_of_mass is not None:
        lam = com_to_lambda(center_of_mass)

    if not 0 < lam < 1:
        raise ValueError("lambda must be between 0 and 1")

    var = returns.ewm(alpha=(1 - lam), adjust=False, min_periods=min_periods).var(bias=False)
    vol = var.pow(0.5)
    if annualize:
        vol = vol * np.sqrt(periods_per_year)
    return vol.fillna(0.0)


def rolling_volatility(
    returns: pd.DataFrame,
    window: int = 63,
    *,
    min_periods: int | None = None,
    annualize: bool = True,
    periods_per_year: int = 252,
) -> pd.DataFrame:
    """Rolling sample standard deviation of returns (ddof=1).

    Matches the EWMA estimator, so switching ``vol_model`` does not change
    position size through a ddof mismatch.
    """

    if window <= 1:
        raise ValueError("window must be greater than one")

    min_periods = min_periods or window
    vol = returns.rolling(window=window, min_periods=min_periods).std(ddof=1)
    if annualize:
        vol = vol * np.sqrt(periods_per_year)
    return vol.fillna(0.0)


def average_true_range(
    high: pd.DataFrame,
    low: pd.DataFrame,
    close: pd.DataFrame,
    window: int = 14,
    *,
    min_periods: int | None = None,
) -> pd.DataFrame:
    """Average True Range (ATR) computed from OHLC bars."""

    if window <= 1:
        raise ValueError("window must be greater than one")

    prev_close = close.shift(1)
    range1 = (high - low).abs()
    range2 = (high - prev_close).abs()
    range3 = (low - prev_close).abs()
    true_range = range1.combine(range2, np.maximum, fill_value=0.0)
    true_range = true_range.combine(range3, np.maximum, fill_value=0.0)

    min_periods = min_periods or window
    return true_range.rolling(window=window, min_periods=min_periods).mean().fillna(0.0)
