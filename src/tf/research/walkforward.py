"""Walk-forward utilities for robustness testing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, TYPE_CHECKING

import pandas as pd

from ..eval.metrics import performance_summary

if TYPE_CHECKING:  # pragma: no cover - import guard for type checking only
    from ..engine.backtester import BacktestResults


@dataclass(frozen=True)
class WalkForwardWindow:
    """Container describing the train and test window for a walk-forward run."""

    insample_start: pd.Timestamp
    insample_end: pd.Timestamp
    oos_start: pd.Timestamp
    oos_end: pd.Timestamp

    @property
    def train_range(self) -> tuple[pd.Timestamp, pd.Timestamp]:
        return (self.insample_start, self.insample_end)

    @property
    def test_range(self) -> tuple[pd.Timestamp, pd.Timestamp]:
        return (self.oos_start, self.oos_end)


@dataclass
class WalkForwardResult:
    """Wrap a backtest result with in-sample/out-of-sample metadata."""

    window: WalkForwardWindow
    backtest: "BacktestResults"
    #: Annualisation basis for the summaries; must match the basis the
    #: strategy was sized on (365 for a 7-day crypto calendar).
    periods_per_year: int = 252

    @property
    def insample_nav(self) -> pd.Series:
        return self.backtest.nav.loc[self.window.insample_start : self.window.insample_end]

    @property
    def oos_nav(self) -> pd.Series:
        return self.backtest.nav.loc[self.window.oos_start : self.window.oos_end]

    @property
    def insample_summary(self) -> dict[str, float]:
        return performance_summary(
            self.insample_nav, periods_per_year=self.periods_per_year
        )

    @property
    def oos_summary(self) -> dict[str, float]:
        return performance_summary(
            self.oos_nav, periods_per_year=self.periods_per_year
        )


def _resolve_index(index: Iterable[pd.Timestamp]) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(index).sort_values().unique()
    if idx.empty:
        raise ValueError("Cannot create walk-forward splits from an empty index")
    return idx


def generate_walk_forward_windows(
    index: Iterable[pd.Timestamp],
    *,
    insample: int,
    oos: int,
    step: int | None = None,
    anchored: bool = True,
) -> list[WalkForwardWindow]:
    """Return walk-forward windows using trading-day counts.

    Parameters
    ----------
    index:
        Ordered collection of timestamps covering the available history.
    insample:
        Number of trading days to allocate to the training window.
    oos:
        Number of trading days in each out-of-sample evaluation window.
    step:
        Number of trading days to roll the window forward.  Defaults to ``oos``.
    anchored:
        When ``True`` the in-sample window is anchored to the first date.
    """

    if insample <= 0 or oos <= 0:
        raise ValueError("insample and oos windows must be positive integers")

    idx = _resolve_index(index)
    total = len(idx)

    step = step or oos
    if step <= 0:
        raise ValueError("step must be a positive integer")

    windows: list[WalkForwardWindow] = []
    start_idx = 0

    while True:
        train_end_idx = start_idx + insample - 1
        test_end_idx = train_end_idx + oos
        if test_end_idx >= total:
            break

        insample_start_idx = 0 if anchored else start_idx
        window = WalkForwardWindow(
            insample_start=idx[insample_start_idx],
            insample_end=idx[train_end_idx],
            oos_start=idx[train_end_idx + 1],
            oos_end=idx[test_end_idx],
        )
        windows.append(window)
        start_idx += step
        if start_idx + insample >= total:
            break

    return windows


def windows_to_splits(windows: Sequence[WalkForwardWindow]) -> list[dict[str, str]]:
    """Return mapping objects suitable for ``Backtester.run`` overrides."""

    splits: list[dict[str, str]] = []
    for window in windows:
        splits.append(
            {
                "start": str(window.insample_start.date()),
                "end": str(window.oos_end.date()),
            }
        )
    return splits
