"""Point-in-time universe eligibility.

Running today's leading cryptocurrencies backwards through history is
survivorship bias: the coins that survived are known only in hindsight. This
module builds a time-indexed mask of which instruments were eligible on each
date, evaluated only from data available at that date, so the backtester cannot
hold an instrument the strategy could not have known about.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import pandas as pd

#: How often eligibility is re-evaluated. Between evaluations membership holds,
#: which is how a real rules-based universe behaves.
EVALUATION_FREQUENCIES = {
    "daily": "D",
    "weekly": "W-MON",
    "monthly": "MS",
    "quarterly": "QS",
}


@dataclass(frozen=True)
class EligibilityRule:
    """Rules deciding whether an instrument may be traded on a given date.

    Parameters
    ----------
    min_history:
        Observations an instrument must already have before it can be traded.
    max_stale_days:
        Consecutive missing observations that make an instrument ineligible.
    evaluation_frequency:
        How often membership is reconsidered.
    entry_lag:
        Additional observations an instrument must wait after first qualifying.
        This models the delay between a rule firing and a position being taken.
    min_adv_usd:
        Optional minimum trailing average daily volume in USD.
    """

    min_history: int = 365
    max_stale_days: int = 1
    evaluation_frequency: str = "monthly"
    entry_lag: int = 30
    min_adv_usd: float | None = None

    def __post_init__(self) -> None:
        if self.min_history < 0:
            raise ValueError("min_history must not be negative")
        if self.entry_lag < 0:
            raise ValueError("entry_lag must not be negative")
        if self.max_stale_days < 0:
            raise ValueError("max_stale_days must not be negative")
        if self.evaluation_frequency not in EVALUATION_FREQUENCIES:
            raise ValueError(
                f"Unknown evaluation_frequency: {self.evaluation_frequency!r}. "
                f"Expected one of {sorted(EVALUATION_FREQUENCIES)}."
            )

    @classmethod
    def from_config(cls, payload: Mapping[str, object] | None) -> "EligibilityRule":
        """Build a rule from a config block, accepting duration strings."""

        if not payload:
            return cls()
        data = dict(payload)
        return cls(
            min_history=_as_bars(data.get("min_history", 365)),
            max_stale_days=_as_bars(data.get("max_stale_days", 1)),
            evaluation_frequency=str(data.get("evaluation_frequency", "monthly")),
            entry_lag=_as_bars(data.get("entry_lag", 30)),
            min_adv_usd=(
                float(data["min_adv_usd"]) if data.get("min_adv_usd") is not None else None
            ),
        )


def _as_bars(value: object) -> int:
    """Coerce ``30`` or ``"30D"`` to a bar count."""

    if isinstance(value, str):
        text = value.strip()
        if text.endswith(("D", "d")):
            text = text[:-1]
        return int(float(text))
    return int(value)


def build_eligibility_mask(
    prices: pd.DataFrame,
    rule: EligibilityRule | None = None,
    *,
    listing_dates: Mapping[str, object] | None = None,
    volumes: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Return a boolean frame of which instruments are tradeable on each date.

    Every input is evaluated as of the date in question. The mask is held
    constant between evaluation dates.
    """

    rule = rule or EligibilityRule()
    if prices.empty:
        return pd.DataFrame(index=prices.index, columns=prices.columns, dtype=bool)

    observed = prices.notna()

    # History accumulated strictly before the current bar, so an instrument is
    # never credited with the observation it is being evaluated on.
    history = observed.cumsum().shift(1).fillna(0.0)
    has_history = history >= rule.min_history

    # Staleness: consecutive bars without a fresh observation.
    stale = _consecutive_missing(observed)
    is_fresh = stale <= rule.max_stale_days

    eligible = has_history & is_fresh

    if listing_dates:
        listed = pd.DataFrame(True, index=prices.index, columns=prices.columns)
        for symbol, listing in listing_dates.items():
            if symbol not in listed.columns or listing is None:
                continue
            listed[symbol] = prices.index >= pd.Timestamp(listing).normalize()
        eligible &= listed

    if rule.min_adv_usd is not None and volumes is not None:
        trailing = volumes.reindex_like(prices).rolling(30, min_periods=1).mean().shift(1)
        eligible &= trailing >= rule.min_adv_usd

    # Entry lag: the rule must have held for this many bars before trading.
    if rule.entry_lag:
        sustained = (
            eligible.rolling(rule.entry_lag + 1, min_periods=rule.entry_lag + 1)
            .min()
            .fillna(0.0)
            .astype(bool)
        )
        eligible = sustained

    return _hold_between_evaluations(eligible, rule.evaluation_frequency)


def _consecutive_missing(observed: pd.DataFrame) -> pd.DataFrame:
    """Return the run length of consecutive missing observations per column."""

    missing = ~observed
    result = pd.DataFrame(index=observed.index, columns=observed.columns, dtype=float)
    for column in observed.columns:
        # cumsum of the observed flag labels each run of missing bars, so a
        # cumulative count within the label is the run length.
        run_id = observed[column].cumsum()
        result[column] = missing[column].groupby(run_id).cumsum()
    return result


def _hold_between_evaluations(eligible: pd.DataFrame, frequency: str) -> pd.DataFrame:
    """Sample eligibility on evaluation dates and hold it in between."""

    if frequency == "daily":
        return eligible

    freq = EVALUATION_FREQUENCIES[frequency]
    evaluation_dates = eligible.resample(freq).first().index
    evaluation_dates = evaluation_dates[evaluation_dates.isin(eligible.index)]
    if len(evaluation_dates) == 0:
        return eligible

    is_evaluation = pd.Series(
        eligible.index.isin(evaluation_dates), index=eligible.index
    )
    held = eligible.where(is_evaluation, other=pd.NA)
    return held.ffill().fillna(False).astype(bool)


def entry_dates(mask: pd.DataFrame) -> dict[str, pd.Timestamp | None]:
    """Return the first date each instrument became eligible."""

    entries: dict[str, pd.Timestamp | None] = {}
    for column in mask.columns:
        active = mask.index[mask[column].to_numpy()]
        entries[column] = active[0] if len(active) else None
    return entries


def eligibility_summary(mask: pd.DataFrame) -> pd.DataFrame:
    """Summarise when each instrument entered and how long it stayed eligible."""

    rows = []
    for column in mask.columns:
        series = mask[column]
        active = mask.index[series.to_numpy()]
        rows.append(
            {
                "symbol": column,
                "entry_date": active[0] if len(active) else pd.NaT,
                "eligible_bars": int(series.sum()),
                "eligible_fraction": float(series.mean()),
            }
        )
    return pd.DataFrame(rows).set_index("symbol")


def thin_universe_fraction(mask: pd.DataFrame, minimum: int = 3) -> float:
    """Return the fraction of dates with fewer than ``minimum`` instruments.

    A diversified crypto backtest is not diversified for most of its history;
    reporting this stops that being invisible.
    """

    if mask.empty:
        return 0.0
    counts = mask.sum(axis=1)
    return float((counts < minimum).mean())
