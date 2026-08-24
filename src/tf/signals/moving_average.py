"""Moving-average based signals."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .normalizers import apply_guardrails, lag_signal


def price_vs_sma(
    prices: pd.DataFrame,
    fast: int = 50,
    slow: int = 200,
    strength_clip: float = 3.0,
    lag: int = 1,
) -> pd.DataFrame:
    """Return the strength of a fast SMA relative to a slow SMA.

    Despite the name this compares two moving averages rather than price to a
    moving average, making it a near-duplicate of
    :func:`moving_average_crossover` (the two differ only in the denominator).
    For price minus a moving average, the PMAC family used in trend-following
    research, use :func:`price_minus_ma`. The name is kept for compatibility.
    """

    if fast <= 0 or slow <= 0:
        raise ValueError("Moving average windows must be positive")
    if fast >= slow:
        raise ValueError("Fast window must be shorter than slow window")

    sma_fast = prices.rolling(window=fast, min_periods=fast).mean()
    sma_slow = prices.rolling(window=slow, min_periods=slow).mean()
    raw = (sma_fast - sma_slow) / (sma_slow.abs() + 1e-9)
    normalized = apply_guardrails(raw, clip=strength_clip)
    return lag_signal(normalized, lag).fillna(0.0)


def moving_average_crossover(
    prices: pd.DataFrame,
    fast: int = 50,
    slow: int = 200,
    strength_clip: float = 3.0,
    lag: int = 1,
) -> pd.DataFrame:
    """Return SMA crossover signal scaled to [-1, 1]."""

    if fast <= 0 or slow <= 0:
        raise ValueError("Moving average windows must be positive")
    if fast >= slow:
        raise ValueError("Fast window must be shorter than slow window")

    ma_fast = prices.rolling(window=fast, min_periods=fast).mean()
    ma_slow = prices.rolling(window=slow, min_periods=slow).mean()
    raw = (ma_fast / (ma_slow + 1e-9)) - 1.0
    normalized = apply_guardrails(raw, clip=strength_clip)
    return lag_signal(normalized, lag).fillna(0.0)


def price_minus_ma(
    prices: pd.DataFrame,
    window: int = 200,
    strength_clip: float = 3.0,
    lag: int = 1,
    transform: str = "tanh",
) -> pd.DataFrame:
    """Return price minus its own moving average (the PMAC family).

    Positive when price trades above its ``window``-bar moving average. This is
    the price-versus-moving-average system named in CTA replication research,
    and unlike :func:`price_vs_sma` it genuinely uses the price level.

    ``transform`` of ``"sign"`` yields a binary signal in {-1, 0, +1}; ``"tanh"``
    yields a continuous strength.
    """

    if window <= 0:
        raise ValueError("Moving average window must be positive")

    ma = prices.rolling(window=window, min_periods=window).mean()
    raw = (prices - ma) / (ma.abs() + 1e-9)

    if transform == "sign":
        signal = raw.apply(np.sign)
    elif transform == "tanh":
        signal = apply_guardrails(raw, clip=strength_clip)
    else:
        raise ValueError(f"Unknown transform: {transform!r}. Expected 'sign' or 'tanh'.")
    return lag_signal(signal, lag).fillna(0.0)
