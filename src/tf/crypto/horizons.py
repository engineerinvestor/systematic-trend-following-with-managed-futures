"""Resolve calendar-duration horizons to bar counts.

Trend-following horizons are quoted as durations ("12 months", "200 days"), but
the signal functions take a number of bars. On a weekday calendar a year is 252
bars; on a 7-day crypto calendar it is 365. Writing horizons as durations and
resolving them against the instrument's calendar keeps a "12-month" signal
twelve months long on either.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import pandas as pd

#: Bars per calendar year, by calendar frequency.
BARS_PER_YEAR = {"weekday": 252, "daily": 365}


def bars_per_year(freq: str = "daily") -> int:
    """Return the number of bars in a year on ``freq``."""

    try:
        return BARS_PER_YEAR[freq]
    except KeyError as exc:
        raise ValueError(
            f"Unknown calendar frequency: {freq!r}. Expected one of {sorted(BARS_PER_YEAR)}."
        ) from exc


def resolve_horizon(horizon: str | int, freq: str = "daily") -> int:
    """Resolve one horizon to a bar count.

    Integers pass through as bar counts. Strings are parsed as pandas offsets,
    so ``"30D"``, ``"3M"``, and ``"1Y"`` all work, and are converted using the
    calendar's bars-per-year.
    """

    if isinstance(horizon, (int,)) and not isinstance(horizon, bool):
        if horizon <= 0:
            raise ValueError(f"Horizon must be positive, got {horizon}")
        return int(horizon)

    text = str(horizon).strip()
    if text.isdigit():
        return int(text)

    try:
        delta = pd.Timedelta(text)
    except ValueError:
        try:
            offset = pd.tseries.frequencies.to_offset(text)
            delta = pd.Timedelta(offset.nanos, unit="ns")
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Could not parse horizon {horizon!r}") from exc

    days = delta.total_seconds() / 86_400.0
    if days <= 0:
        raise ValueError(f"Horizon must be positive, got {horizon!r}")

    bars = round(days * bars_per_year(freq) / 365.0)
    return max(int(bars), 1)


def resolve_horizons(
    horizons: Iterable[str | int], freq: str = "daily"
) -> tuple[int, ...]:
    """Resolve a sequence of horizons to bar counts, preserving order."""

    resolved = tuple(resolve_horizon(h, freq) for h in horizons)
    if not resolved:
        raise ValueError("At least one horizon is required")
    return resolved


#: The MOP time-series momentum horizons, as durations.
TSMOM_HORIZONS: Sequence[str] = ("30D", "90D", "365D")

#: Thirteen horizons spanning one week to one year, denser at the short end.
#: CTA replication research uses a ladder of roughly this shape; the specific
#: day values are this package's default rather than any manager's constants.
REPLICATION_HORIZONS: Sequence[int] = (
    5, 10, 15, 20, 30, 40, 60, 90, 120, 150, 180, 220, 260,
)
