"""Momentum style signals."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from .normalizers import apply_guardrails, lag_signal


def timeseries_momentum(
    prices: pd.DataFrame,
    lookbacks: Sequence[int] = (63, 126, 252),
    skip_last_n: int = 20,
    strength_clip: float = 3.0,
    lag: int = 1,
    transform: str = "tanh",
    weighting: str = "inv_sqrt",
) -> pd.DataFrame:
    """Compute multi-horizon time-series momentum strength.

    ``skip_last_n`` omits the most recent observations, the skip-month
    convention from cross-sectional momentum. Time-series momentum as defined by
    Moskowitz, Ooi, and Pedersen uses no skip, so pass ``skip_last_n=0`` when
    replicating it.

    ``transform`` of ``"sign"`` takes the sign of each horizon's return before
    combining, which is the published TSMOM construction; ``"tanh"`` keeps the
    continuous strength this package used previously.

    ``weighting`` of ``"equal"`` weights horizons equally, as the 1/3/12-month
    ensemble does; ``"inv_sqrt"`` weights by one over the square root of the
    lookback.
    """

    if prices.empty:
        raise ValueError("Price history is empty")

    if any(lb <= 0 for lb in lookbacks):
        raise ValueError("Lookbacks must be positive integers")

    if transform not in {"tanh", "sign"}:
        raise ValueError(f"Unknown transform: {transform!r}. Expected 'sign' or 'tanh'.")
    if weighting == "equal":
        weights = np.ones(len(lookbacks), dtype=float)
    elif weighting == "inv_sqrt":
        weights = np.array([1 / np.sqrt(lb) for lb in lookbacks], dtype=float)
    else:
        raise ValueError(
            f"Unknown weighting: {weighting!r}. Expected 'equal' or 'inv_sqrt'."
        )
    weights = weights / weights.sum()

    prices = prices.sort_index()
    combined = pd.DataFrame(index=prices.index, columns=prices.columns, dtype=float)
    shifted = prices.shift(skip_last_n) if skip_last_n else prices

    for weight, lookback in zip(weights, lookbacks):
        momentum = shifted / shifted.shift(lookback) - 1.0
        if transform == "sign":
            # Take the sign per horizon, then average, so the combined signal is
            # the net agreement across horizons rather than a return magnitude.
            momentum = momentum.apply(np.sign)
        combined = combined.add(momentum * weight, fill_value=0.0)

    if transform == "sign":
        return lag_signal(combined, lag).fillna(0.0)
    normalized = apply_guardrails(combined, clip=strength_clip)
    return lag_signal(normalized, lag).fillna(0.0)
