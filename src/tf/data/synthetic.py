"""Synthetic price generation for smoke tests and offline examples."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd


def generate_synthetic_prices(
    symbols: Sequence[str],
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    seed: int | None = 42,
    *,
    freq: str = "C",
    vol: float | Mapping[str, float] | None = None,
    mu: float | Mapping[str, float] | None = None,
    common_factor: float = 0.0,
    start_offsets: Mapping[str, int] | None = None,
) -> pd.DataFrame:
    """Generate mildly trending random walks.

    ``freq`` is passed to the date range: ``"C"`` gives weekday sessions and
    ``"D"`` gives every calendar day, matching :class:`~tf.data.calendar.TradingCalendar`.

    ``vol`` and ``mu`` set daily volatility and drift, either globally or per
    symbol; without them the original heteroskedastic defaults apply.

    ``common_factor`` in ``[0, 1]`` mixes a shared market factor into every
    symbol. The default of 0 leaves series independent, which is optimistic:
    uncorrelated assets flatter portfolio volatility targeting, and crypto pairs
    in particular are strongly correlated.

    ``start_offsets`` maps a symbol to the number of leading observations left as
    NaN, so a universe can be given staggered listing histories.
    """

    rng = np.random.default_rng(seed)
    if freq.upper() == "D":
        dates = pd.date_range(start, end, freq="D")
    else:
        dates = pd.bdate_range(start, end, freq=freq)

    if not 0.0 <= common_factor <= 1.0:
        raise ValueError("common_factor must lie in [0, 1]")

    market = rng.normal(0.0, 1.0, size=len(dates)) if common_factor > 0 else None

    data = {}
    for i, sym in enumerate(symbols):
        mu_daily = _resolve(mu, sym, 0.0002 + 0.0001 * np.sin(i))
        vol_daily = _resolve(vol, sym, 0.01 + 0.002 * (i % 3))

        idiosyncratic = rng.normal(0.0, 1.0, size=len(dates))
        if market is not None:
            shocks_std = (
                np.sqrt(common_factor) * market
                + np.sqrt(1.0 - common_factor) * idiosyncratic
            )
        else:
            shocks_std = idiosyncratic
        shocks = mu_daily + vol_daily * shocks_std

        # Inject a few trend regimes so trend systems have something to find.
        for k in range(50, len(shocks), 500):
            shocks[k : k + 100] += 0.0008 * np.sign(np.sin(k + i))

        levels = 100 * np.exp(np.cumsum(shocks))
        series = pd.Series(levels, index=dates)

        offset = int((start_offsets or {}).get(sym, 0))
        if offset > 0:
            series.iloc[:offset] = np.nan
        data[sym] = series

    return pd.DataFrame(data, index=dates)


def _resolve(
    value: float | Mapping[str, float] | None, symbol: str, default: float
) -> float:
    if value is None:
        return default
    if isinstance(value, Mapping):
        return float(value.get(symbol, default))
    return float(value)
