"""Trading calendar utilities for aligning daily price data."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

import pandas as pd

logger = logging.getLogger(__name__)


def _normalize_holidays(holidays: Optional[Iterable[pd.Timestamp]]) -> list[pd.Timestamp]:
    if not holidays:
        return []
    normalized: list[pd.Timestamp] = []
    for holiday in holidays:
        ts = pd.Timestamp(holiday).normalize()
        if ts not in normalized:
            normalized.append(ts)
    return normalized


#: Calendar names accepted in ``config["data"]["calendar"]``, mapped to a session
#: frequency. Names are matched case-insensitively.
_NAMED_FREQUENCIES = {
    "generic": "weekday",
    "weekday": "weekday",
    "daily": "daily",
    "crypto": "daily",
    "crypto_daily": "daily",
    "24/7": "daily",
}

_VALID_FREQUENCIES = ("weekday", "daily")


@dataclass(slots=True)
class TradingCalendar:
    """Trading calendar with optional holiday exclusions.

    ``freq`` selects the session frequency. ``"weekday"`` yields Monday to Friday
    sessions and suits exchange-traded futures. ``"daily"`` yields every calendar
    day and suits cryptocurrency, which trades through weekends; aligning crypto
    to a weekday calendar would silently discard two sevenths of its observations.
    """

    name: str = "weekday"
    holidays: Sequence[pd.Timestamp] = field(default_factory=list)
    freq: str = "weekday"

    def __post_init__(self) -> None:
        self.holidays = tuple(_normalize_holidays(self.holidays))
        freq = str(self.freq).lower()
        if freq not in _VALID_FREQUENCIES:
            raise ValueError(
                f"Unknown calendar frequency: {self.freq!r}. "
                f"Expected one of {_VALID_FREQUENCIES}."
            )
        self.freq = freq

    @classmethod
    def from_name(
        cls,
        name: str | None,
        *,
        holidays: Optional[Iterable[pd.Timestamp]] = None,
    ) -> "TradingCalendar":
        """Build a calendar from a config name such as ``"CRYPTO_DAILY"``.

        ``None`` or an unrecognised name yields the weekday default, so configs
        written before the crypto extension keep their behaviour.
        """

        if name is None:
            return cls(holidays=list(holidays or []))
        key = str(name).strip().lower()
        freq = _NAMED_FREQUENCIES.get(key)
        if freq is None:
            logger.warning(
                "Unknown calendar name %r; defaulting to a weekday calendar", name
            )
            freq = "weekday"
        return cls(name=key, holidays=list(holidays or []), freq=freq)

    def sessions(self, start: str | pd.Timestamp, end: str | pd.Timestamp) -> pd.DatetimeIndex:
        """Return trading sessions between ``start`` and ``end`` (inclusive)."""

        if self.freq == "daily":
            sessions = pd.date_range(start=start, end=end, freq="D")
        else:
            sessions = pd.bdate_range(start=start, end=end, freq="C")
        if not self.holidays:
            return sessions
        mask = ~sessions.isin(self.holidays)
        return sessions[mask]

    def align(self, frame: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
        """Reindex price data to the calendar and forward-fill gaps."""

        if not isinstance(frame.index, pd.DatetimeIndex):
            raise TypeError("Expected price data indexed by pandas.DatetimeIndex")

        target_index = self.sessions(start, end)
        aligned = frame.reindex(target_index)
        missing = aligned.isna().sum().sum()
        if missing:
            logger.debug("Forward filling %s missing data points after calendar align", missing)
        return aligned.ffill()

    def validate(self, frame: pd.DataFrame) -> None:
        """Run basic sanity checks on a price dataframe."""

        index = frame.index
        if not isinstance(index, pd.DatetimeIndex):  # pragma: no cover - defensive
            raise TypeError("Price data must be indexed by pandas.DatetimeIndex")
        if not index.is_monotonic_increasing:
            raise ValueError("Price data index must be sorted in increasing order")
        if index.has_duplicates:
            duplicates = index[index.duplicated()].strftime("%Y-%m-%d").tolist()
            raise ValueError(f"Price data contains duplicate dates: {duplicates}")

