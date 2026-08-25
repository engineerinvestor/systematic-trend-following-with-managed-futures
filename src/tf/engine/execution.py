"""Execution helpers for simulating order fills and contract rolls."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


@dataclass
class Order:
    """Simple order object stored in the execution queue."""

    order_id: int
    submitted: pd.Timestamp
    symbol: str
    qty: float
    reason: str = "rebalance"
    target: float | None = None
    initial_qty: float = field(init=False)

    def __post_init__(self) -> None:  # pragma: no cover - trivial
        self.qty = float(self.qty)
        self.initial_qty = float(self.qty)


class RollEngine:
    """Generate roll orders from a simple schedule mapping."""

    def __init__(self, schedule: Mapping[str, Sequence[str | pd.Timestamp]] | None) -> None:
        normalized: dict[pd.Timestamp, set[str]] = {}
        if schedule:
            for symbol, dates in schedule.items():
                for dt in dates:
                    ts = pd.Timestamp(dt).normalize()
                    normalized.setdefault(ts, set()).add(symbol)
        self._schedule = normalized

    def orders_for(self, submit_ts: pd.Timestamp, positions: pd.Series) -> Iterable[tuple[str, float]]:
        if not self._schedule:
            return []
        day = pd.Timestamp(submit_ts).normalize()
        symbols = self._schedule.get(day)
        if not symbols:
            return []
        orders: list[tuple[str, float]] = []
        for sym in symbols:
            qty = float(positions.get(sym, 0.0))
            if abs(qty) <= 1e-8:
                continue
            # Exit the front contract and re-enter the next one.
            orders.append((sym, -qty))
            orders.append((sym, qty))
        return orders


def resolve_adv_frame(
    adv_source: float | Mapping[str, float] | pd.DataFrame | None,
    index: pd.Index,
    symbols: Sequence[str],
    *,
    default: float,
) -> pd.DataFrame:
    """Return a dataframe of ADV values aligned with price history."""

    if isinstance(adv_source, pd.DataFrame):
        frame = adv_source.reindex(index=index, columns=symbols)
        return frame.ffill().bfill().fillna(default)

    if isinstance(adv_source, Mapping):
        data = {}
        for sym in symbols:
            value = float(adv_source.get(sym, default))
            data[sym] = value if np.isfinite(value) else float(default)
        return pd.DataFrame(data, index=index, dtype=float)

    if isinstance(adv_source, (int, float)):
        return pd.DataFrame(float(adv_source), index=index, columns=symbols, dtype=float)

    return pd.DataFrame(float(default), index=index, columns=symbols, dtype=float)


def participation_slippage(
    *,
    qty: float,
    adv: float,
    k: float,
    alpha: float,
    tick_value: float,
    min_ticks: float,
) -> float:
    """Return a simple participation style slippage estimate."""

    qty = float(qty)
    if qty == 0:
        return 0.0
    adv = float(adv)
    if not np.isfinite(adv) or adv <= 0:
        # NaN previously slid through two comparisons and collapsed slippage to
        # the floor; a missing ADV must cost like zero liquidity, not infinite.
        ticks = max(min_ticks, k)
    else:
        participation = min(abs(qty) / adv, 1.0)
        ticks = max(min_ticks, k * participation**alpha)
    return abs(qty) * ticks * tick_value

