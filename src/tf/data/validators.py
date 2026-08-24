"""Validation helpers for market data tables."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class SuspensionSpan:
    """Description of a trading suspension detected in the data."""

    start: pd.Timestamp
    end: pd.Timestamp
    length: int


def validate_price_data(
    frame: pd.DataFrame,
    *,
    min_price: float | None = 0.0,
    max_consecutive_missing: int = 3,
    allow_leading_gaps: bool = True,
) -> None:
    """Raise if the dataframe contains obvious data quality issues.

    ``allow_leading_gaps`` ignores missing observations before an instrument's
    first price. In a universe with staggered listings those leading NaNs mean
    the instrument did not exist yet, which is not a data-quality problem; a gap
    inside its observed history still is.
    """

    if frame.empty:
        raise ValueError("Price dataframe is empty")

    if not isinstance(frame.index, pd.DatetimeIndex):
        raise TypeError("Price data must use a pandas.DatetimeIndex")

    if frame.index.has_duplicates:
        dupes = frame.index[frame.index.duplicated()].strftime("%Y-%m-%d").tolist()
        raise ValueError(f"Price data contains duplicate dates: {dupes}")

    if min_price is not None:
        if (frame <= min_price).any().any():
            raise ValueError("Detected prices at or below the configured minimum")

    missing_mask = frame.isna()
    if not missing_mask.values.any():
        return

    if max_consecutive_missing < 0:
        return

    for column in frame.columns:
        mask = missing_mask[column]
        if allow_leading_gaps:
            observed = ~mask
            if not observed.any():
                raise ValueError(f"Column '{column}' has no observations at all")
            first_observation = observed.idxmax()
            mask = mask.loc[first_observation:]
        _ensure_gap_limit(column, mask, max_consecutive_missing)


def detect_trading_suspensions(
    frame: pd.DataFrame, *,
    min_gap: int = 3,
) -> dict[str, list[SuspensionSpan]]:
    """Return spans of consecutive missing observations per instrument.

    Parameters
    ----------
    frame:
        Wide dataframe containing one column per instrument.
    min_gap:
        Minimum number of consecutive missing observations that should be
        considered a suspension.  Set to ``0`` to return all gaps.
    """

    if frame.empty:
        return {}

    suspensions: dict[str, list[SuspensionSpan]] = {}
    for column in frame.columns:
        spans = list(_iter_missing_spans(frame[column].isna()))
        filtered = [span for span in spans if span.length > min_gap]
        if filtered:
            suspensions[column] = filtered
    return suspensions


def detect_limit_moves(
    frame: pd.DataFrame,
    *,
    threshold: float = 0.07,
) -> pd.DataFrame:
    """Return a boolean mask with limit-up/down candidates.

    The helper flags days where the absolute percentage change exceeds the
    provided ``threshold``.  The caller can join this mask back to the raw
    dataframe to run manual investigations or downstream filters.
    """

    if frame.empty:
        return pd.DataFrame(columns=frame.columns, index=frame.index, dtype=bool)
    if threshold <= 0:
        raise ValueError("Limit threshold must be strictly positive")

    returns = frame.pct_change()
    flagged = returns.abs() > float(threshold)
    if flagged.empty:
        return pd.DataFrame(columns=frame.columns, index=frame.index, dtype=bool)
    return flagged.loc[flagged.any(axis=1)]


def _iter_missing_spans(mask: pd.Series) -> Iterable[SuspensionSpan]:
    if not isinstance(mask, pd.Series):
        raise TypeError("Missing mask must be a pandas Series")

    run_length = 0
    run_start: pd.Timestamp | None = None
    last_ts: pd.Timestamp | None = None
    for ts, missing in mask.items():
        if bool(missing):
            run_length += 1
            if run_start is None:
                run_start = ts
            last_ts = ts
        elif run_length:
            yield SuspensionSpan(start=run_start, end=last_ts, length=run_length)
            run_length = 0
            run_start = None
            last_ts = None

    if run_length and run_start is not None and last_ts is not None:
        yield SuspensionSpan(start=run_start, end=last_ts, length=run_length)


def _ensure_gap_limit(name: str, mask: pd.Series, max_consecutive: int) -> None:
    if max_consecutive < 0:
        return

    for span in _iter_missing_spans(mask):
        if span.length > max_consecutive:
            raise ValueError(
                f"Column '{name}' has a gap of {span.length} consecutive missing observations"
            )
