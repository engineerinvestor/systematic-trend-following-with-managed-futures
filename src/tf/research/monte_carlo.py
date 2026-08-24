"""Monte Carlo utilities via block bootstrap resampling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from ..eval.metrics import TRADING_DAYS


@dataclass(frozen=True)
class BootstrapInterval:
    """Container summarising bootstrap estimates for a metric."""

    mean: float
    lower: float
    upper: float


def _moving_block_bootstrap(
    returns: np.ndarray,
    *,
    block_size: int,
    n_samples: int,
    random_state: np.random.Generator,
) -> np.ndarray:
    """Return bootstrap sample matrix with shape (n_samples, len(returns))."""

    n = len(returns)
    if n == 0:
        raise ValueError("Cannot bootstrap an empty return series")
    if block_size <= 0:
        raise ValueError("block_size must be positive")

    block_size = min(block_size, n)
    blocks = np.array([returns[i : i + block_size] for i in range(n - block_size + 1)], dtype=float)
    n_blocks = len(blocks)
    if n_blocks == 0:
        blocks = returns.reshape(1, -1)
        n_blocks = 1

    samples = np.empty((n_samples, n), dtype=float)
    for i in range(n_samples):
        draw = random_state.integers(0, n_blocks, size=int(np.ceil(n / block_size)))
        stitched = np.concatenate(blocks[draw])
        samples[i, :] = stitched[:n]
    return samples


def _sharpe_ratio(sample: np.ndarray, periods_per_year: int = TRADING_DAYS) -> float:
    mean = sample.mean()
    std = sample.std(ddof=1)
    if std == 0:
        return 0.0
    return float(np.sqrt(periods_per_year) * mean / std)


def _max_drawdown(sample: np.ndarray, periods_per_year: int = TRADING_DAYS) -> float:
    nav = np.cumprod(1 + sample)
    running_max = np.maximum.accumulate(nav)
    drawdowns = nav / running_max - 1.0
    return float(drawdowns.min())


def _annualised_return(sample: np.ndarray, periods_per_year: int = TRADING_DAYS) -> float:
    return float(sample.mean() * periods_per_year)


def _annualised_vol(sample: np.ndarray, periods_per_year: int = TRADING_DAYS) -> float:
    return float(sample.std(ddof=1) * np.sqrt(periods_per_year))


def _hit_rate(sample: np.ndarray, periods_per_year: int = TRADING_DAYS) -> float:
    return float((sample > 0).mean())


_METRIC_FUNCTIONS: dict[str, Callable[[np.ndarray, int], float]] = {
    "sharpe": _sharpe_ratio,
    "max_drawdown": _max_drawdown,
    "annual_return": _annualised_return,
    "volatility": _annualised_vol,
    "hit_rate": _hit_rate,
}


def bootstrap_confidence_intervals(
    returns: pd.Series | np.ndarray,
    *,
    metrics: list[str] | None = None,
    n_samples: int = 1_000,
    block_size: int = 20,
    ci: float = 0.95,
    seed: int | None = None,
    periods_per_year: int = TRADING_DAYS,
) -> dict[str, BootstrapInterval]:
    """Estimate confidence intervals for key metrics using block bootstrap."""

    if isinstance(returns, pd.Series):
        values = returns.dropna().to_numpy(dtype=float)
    else:
        values = np.asarray(returns, dtype=float)
        values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("returns must contain at least one finite observation")

    metrics = metrics or ["sharpe", "max_drawdown"]
    rng = np.random.default_rng(seed)
    samples = _moving_block_bootstrap(values, block_size=block_size, n_samples=n_samples, random_state=rng)

    alpha = (1.0 - ci) / 2.0
    lower_q = 100 * alpha
    upper_q = 100 * (1 - alpha)

    estimates: dict[str, BootstrapInterval] = {}
    for metric in metrics:
        fn = _METRIC_FUNCTIONS.get(metric.lower())
        if fn is None:
            raise KeyError(f"Unsupported metric '{metric}'. Available: {list(_METRIC_FUNCTIONS)}")
        stats = np.apply_along_axis(fn, 1, samples, periods_per_year)
        interval = BootstrapInterval(
            mean=float(np.mean(stats)),
            lower=float(np.percentile(stats, lower_q)),
            upper=float(np.percentile(stats, upper_q)),
        )
        estimates[metric] = interval

    return estimates
