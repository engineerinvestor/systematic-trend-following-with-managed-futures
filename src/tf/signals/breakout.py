"""Breakout style price signals."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .normalizers import apply_guardrails, lag_signal


def channel_breakout(
    prices: pd.DataFrame,
    window: int = 100,
    strength_clip: float = 3.0,
    lag: int = 1,
) -> pd.DataFrame:
    """Return where price sits inside its recent high-low channel.

    The rolling window includes the current bar, so price is always between the
    window high and low and the raw ratio is confined to [-0.5, +0.5]. This is a
    channel-position oscillator, not a breakout detector: it cannot signal that
    price has broken out of the channel, because by construction it never has.
    For an entry-style Donchian rule use :func:`donchian_breakout`.
    """

    if window <= 1:
        raise ValueError("Breakout window must be greater than 1")

    high = prices.rolling(window=window, min_periods=window).max()
    low = prices.rolling(window=window, min_periods=window).min()
    width = (high - low).replace(0, np.nan)
    mid = (high + low) / 2.0
    raw = (prices - mid) / (width.abs() + 1e-9)
    normalized = apply_guardrails(raw, clip=strength_clip)
    return lag_signal(normalized, lag).fillna(0.0)


def donchian_breakout(
    prices: pd.DataFrame,
    window: int = 100,
    lag: int = 1,
    hold: bool = True,
) -> pd.DataFrame:
    """Return a Donchian breakout signal in {-1, 0, +1}.

    The high and low bands are computed over the ``window`` bars *before* the
    current bar, so a close above the prior high is a genuine breakout. Going
    long on a new high and short on a new low is the classic entry rule.

    With ``hold`` the position persists between breakouts, which is what a
    trend-following system does; without it the signal is non-zero only on the
    bars that break out.
    """

    if window <= 1:
        raise ValueError("Breakout window must be greater than 1")

    prior = prices.shift(1)
    high = prior.rolling(window=window, min_periods=window).max()
    low = prior.rolling(window=window, min_periods=window).min()

    signal = pd.DataFrame(
        np.where(prices > high, 1.0, np.where(prices < low, -1.0, np.nan)),
        index=prices.index,
        columns=prices.columns,
    )
    if hold:
        signal = signal.ffill()
    return lag_signal(signal, lag).fillna(0.0)
