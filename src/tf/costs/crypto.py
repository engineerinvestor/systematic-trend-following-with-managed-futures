"""Cost model for cryptocurrency futures and spot execution.

Costs are stated per instrument and per era rather than as one universe-wide
number, because crypto spreads differ by an order of magnitude across coins and
have tightened substantially as the contracts matured. Every figure here is an
assumption, not a measurement: the point of the falsification suite is to run
the same strategy at multiples of these numbers and see what survives.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import pandas as pd

from ..engine.execution import participation_slippage

#: Commission per contract per side, in USD, for CME crypto futures. Retail
#: all-in rates including clearing and exchange fees are typically in this
#: range; brokers differ, so treat it as a starting point.
DEFAULT_COMMISSION_PER_CONTRACT = 2.50

#: Assumed round-trip half-spread in basis points of notional, by instrument.
#: BTC and ETH quote inside a basis point or two; the newer contracts are wider.
DEFAULT_HALF_SPREAD_BPS: Mapping[str, float] = {
    "BTC": 1.0,
    "ETH": 1.5,
    "SOL": 4.0,
    "XRP": 5.0,
    "ADA": 8.0,
    "LINK": 8.0,
    "XLM": 8.0,
    "AVAX": 10.0,
    "SUI": 10.0,
}

#: Fallback for an instrument with no entry above.
DEFAULT_HALF_SPREAD_FALLBACK_BPS = 10.0

#: Spreads were materially wider in the early years of each listing. Costs
#: before this date are scaled by :data:`EARLY_ERA_MULTIPLIER`.
EARLY_ERA_END = pd.Timestamp("2021-01-01")
EARLY_ERA_MULTIPLIER = 3.0


@dataclass(frozen=True)
class CryptoCostModel:
    """Commission, spread, and impact assumptions for a crypto universe."""

    commission_per_contract: float = DEFAULT_COMMISSION_PER_CONTRACT
    half_spread_bps: Mapping[str, float] | None = None
    impact_k: float = 0.05
    impact_alpha: float = 0.5
    min_slippage_ticks: float = 0.5
    cost_multiplier: float = 1.0

    def half_spread_for(
        self, symbol: str, timestamp: pd.Timestamp | str | None = None
    ) -> float:
        """Return the assumed half-spread in basis points."""

        table = self.half_spread_bps or DEFAULT_HALF_SPREAD_BPS
        bps = float(table.get(symbol.upper(), DEFAULT_HALF_SPREAD_FALLBACK_BPS))
        if timestamp is not None and pd.Timestamp(timestamp) < EARLY_ERA_END:
            bps *= EARLY_ERA_MULTIPLIER
        return bps * float(self.cost_multiplier)

    def spread_cost(
        self,
        *,
        symbol: str,
        notional: float,
        timestamp: pd.Timestamp | str | None = None,
    ) -> float:
        """Return the spread cost of trading ``notional`` USD of ``symbol``."""

        bps = self.half_spread_for(symbol, timestamp)
        return abs(float(notional)) * bps / 10_000.0

    def commission_cost(self, *, quantity: float) -> float:
        """Return the commission for ``quantity`` contracts."""

        return abs(float(quantity)) * self.commission_per_contract * self.cost_multiplier

    def impact_cost(
        self,
        *,
        quantity: float,
        adv: float,
        tick_value: float,
    ) -> float:
        """Return participation impact, reusing the engine's estimator."""

        return participation_slippage(
            qty=quantity,
            adv=adv,
            k=self.impact_k * self.cost_multiplier,
            alpha=self.impact_alpha,
            tick_value=tick_value,
            min_ticks=self.min_slippage_ticks * self.cost_multiplier,
        )

    def total_cost(
        self,
        *,
        symbol: str,
        quantity: float,
        price: float,
        point_value: float,
        adv: float,
        tick_value: float,
        timestamp: pd.Timestamp | str | None = None,
    ) -> float:
        """Return commission plus spread plus impact for a single fill."""

        notional = abs(float(quantity)) * float(price) * float(point_value)
        return (
            self.commission_cost(quantity=quantity)
            + self.spread_cost(symbol=symbol, notional=notional, timestamp=timestamp)
            + self.impact_cost(quantity=quantity, adv=adv, tick_value=tick_value)
        )

    def as_execution_config(self) -> dict:
        """Return an ``execution`` config block matching this model."""

        return {
            "commission_per_contract": self.commission_per_contract,
            "impact": {"k": self.impact_k, "alpha": self.impact_alpha},
            "min_slippage_ticks": self.min_slippage_ticks,
            "cost_multiplier": self.cost_multiplier,
        }


def scale_execution_costs(execution_cfg: Mapping, multiplier: float) -> dict:
    """Return ``execution_cfg`` with every cost component scaled.

    The stress battery varies costs as one knob, but costs live in several
    separate config keys, so this scales them together.
    """

    if multiplier < 0:
        raise ValueError("cost multiplier must not be negative")
    scaled = {k: v for k, v in dict(execution_cfg).items()}
    scaled["commission_per_contract"] = (
        float(scaled.get("commission_per_contract", 0.0)) * multiplier
    )
    scaled["min_slippage_ticks"] = float(scaled.get("min_slippage_ticks", 0.0)) * multiplier
    impact = dict(scaled.get("impact", {}) or {})
    impact["k"] = float(impact.get("k", 0.0)) * multiplier
    scaled["impact"] = impact
    scaled["cost_multiplier"] = multiplier
    return scaled
