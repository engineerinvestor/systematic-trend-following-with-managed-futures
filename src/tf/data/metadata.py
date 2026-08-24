"""Utilities for describing futures contracts and universes.

This module provides light-weight data classes that capture the minimal
metadata required by the starter backtester.  The objects are intentionally
simple – they focus on symbol identity, sector grouping, point value and the
information necessary to locate a tradeable time-series from public data
sources such as Yahoo! Finance.

Future project phases can extend these structures with richer attributes
like tick-size, expiration calendars or exchange specific roll logic, but the
goal here is to give the data layer a well-defined schema that downstream code
can reason about today.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, MutableMapping, Optional, Sequence


@dataclass(frozen=True, slots=True)
class ContractMetadata:
    """Describe a single futures contract or proxy time-series.

    Parameters
    ----------
    symbol:
        Internal symbol used throughout the backtester.
    sector:
        High-level sector bucket used for risk aggregation.
    point_value:
        Monetary value of a one point price move for the contract.
    currency:
        Reporting currency for the point value.  Defaults to USD.
    contract_step:
        Minimum contract increment when rounding target positions.  Defaults to 1.
    data_source:
        Identifier for the preferred market data vendor.  ``"yahoo"`` is
        supported out of the box.
    data_symbol:
        Optional vendor specific symbol override.  When omitted and the data
        source is Yahoo the loader falls back to the common ``"<sym>=F"``
        futures convention.
    description:
        Human readable text used for logs or reports.
    tick_size:
        Minimum price increment.  Execution otherwise falls back to a single
        universe-wide tick value, which is wrong whenever instruments differ.
    listing_date:
        First date the instrument was tradeable.  Point-in-time universe
        eligibility uses this so a backtest cannot hold an instrument that did
        not yet exist.
    calendar:
        Optional per-instrument calendar name, e.g. ``"CRYPTO_DAILY"``.
    venue:
        Exchange or venue identifier, e.g. ``"CME"``.
    contract_size:
        Units of the underlying per contract, where that differs from the
        monetary point value.
    session_regime_changes:
        Dates on which the venue's session calendar changed, as a mapping of
        ISO date to calendar name.  CME crypto futures traded a five-day week
        until 29 May 2026 and around the clock afterwards, so a single backtest
        spans two session regimes.
    """

    symbol: str
    sector: str
    point_value: float
    currency: str = "USD"
    contract_step: float = 1.0
    data_source: str = "yahoo"
    data_symbol: Optional[str] = None
    description: Optional[str] = None
    tick_size: Optional[float] = None
    listing_date: Optional[str] = None
    calendar: Optional[str] = None
    venue: Optional[str] = None
    contract_size: Optional[float] = None
    session_regime_changes: Optional[Mapping[str, str]] = None

    def __post_init__(self) -> None:  # pragma: no cover - dataclass hook
        if not self.symbol:
            raise ValueError("Contract symbol must be provided")
        if self.point_value <= 0:
            raise ValueError("Point value must be positive")
        if not self.sector:
            raise ValueError("Sector must be provided")
        if self.contract_step <= 0:
            raise ValueError("Contract step must be positive")
        if self.tick_size is not None and self.tick_size <= 0:
            raise ValueError("Tick size must be positive when provided")
        if self.contract_size is not None and self.contract_size <= 0:
            raise ValueError("Contract size must be positive when provided")

    @property
    def listing_timestamp(self):
        """Return ``listing_date`` as a Timestamp, or ``None`` when unset."""

        if self.listing_date is None:
            return None
        import pandas as pd

        return pd.Timestamp(self.listing_date).normalize()

    @property
    def vendor_symbol(self) -> str:
        """Return the symbol understood by the configured data vendor."""

        if self.data_symbol:
            return self.data_symbol
        if self.data_source.lower() == "yahoo":
            # Yahoo futures commonly use the ``ES=F`` style suffix.
            if self.symbol.endswith("=F"):
                return self.symbol
            return f"{self.symbol}=F"
        return self.symbol

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "ContractMetadata":
        """Construct metadata from dictionaries (e.g. YAML or JSON records)."""

        data = dict(payload)
        try:
            symbol = str(data.pop("symbol"))
        except KeyError as exc:  # pragma: no cover - defensive
            raise KeyError("Contract dictionary missing 'symbol'") from exc
        try:
            point_value = float(data.pop("point_value"))
        except KeyError as exc:  # pragma: no cover - defensive
            raise KeyError("Contract dictionary missing 'point_value'") from exc
        sector = str(data.pop("sector", ""))
        data_source = str(data.pop("data_source", "yahoo"))
        contract_step = float(data.pop("contract_step", 1.0))
        data_symbol = data.pop("data_symbol", None)
        description = data.pop("description", None)
        currency = str(data.pop("currency", "USD"))
        tick_size = data.pop("tick_size", None)
        listing_date = data.pop("listing_date", None)
        calendar = data.pop("calendar", None)
        venue = data.pop("venue", None)
        contract_size = data.pop("contract_size", None)
        session_regime_changes = data.pop("session_regime_changes", None)
        if data:
            # Surface configuration typos early.
            unknown = ", ".join(sorted(data))
            raise KeyError(f"Unknown contract metadata fields: {unknown}")
        return cls(
            symbol=symbol,
            sector=sector,
            point_value=point_value,
            currency=currency,
            contract_step=contract_step,
            data_source=data_source,
            data_symbol=str(data_symbol) if data_symbol is not None else None,
            description=str(description) if description is not None else None,
            tick_size=float(tick_size) if tick_size is not None else None,
            listing_date=str(listing_date) if listing_date is not None else None,
            calendar=str(calendar) if calendar is not None else None,
            venue=str(venue) if venue is not None else None,
            contract_size=float(contract_size) if contract_size is not None else None,
            session_regime_changes=(
                dict(session_regime_changes)
                if session_regime_changes is not None
                else None
            ),
        )


@dataclass(slots=True)
class UniverseDefinition:
    """Collection of :class:`ContractMetadata` objects with validation helpers."""

    contracts: Sequence[ContractMetadata] = field(default_factory=list)

    def __post_init__(self) -> None:
        symbols = [c.symbol for c in self.contracts]
        if len(symbols) != len(set(symbols)):
            raise ValueError("Universe contains duplicate contract symbols")

    @classmethod
    def from_payload(
        cls, entries: Iterable[Mapping[str, object] | ContractMetadata]
    ) -> "UniverseDefinition":
        """Build a validated universe from dictionaries or metadata objects."""

        contracts: list[ContractMetadata] = []
        for entry in entries:
            if isinstance(entry, ContractMetadata):
                contracts.append(entry)
            else:
                contracts.append(ContractMetadata.from_dict(entry))
        return cls(contracts)

    def as_dataframe(self) -> "pd.DataFrame":
        """Return the universe attributes as a tidy ``pandas`` dataframe."""

        import pandas as pd  # Local import keeps pandas optional for callers

        data = [
            {
                "symbol": c.symbol,
                "sector": c.sector,
                "point_value": c.point_value,
                "currency": c.currency,
                "contract_step": c.contract_step,
                "data_source": c.data_source,
                "data_symbol": c.vendor_symbol,
                "description": c.description or "",
                "tick_size": c.tick_size,
                "listing_date": c.listing_date,
                "calendar": c.calendar,
                "venue": c.venue,
                "contract_size": c.contract_size,
            }
            for c in self.contracts
        ]
        return pd.DataFrame(data).set_index("symbol")

    def by_symbol(self) -> MutableMapping[str, ContractMetadata]:
        """Return a dictionary keyed by internal symbol."""

        return {c.symbol: c for c in self.contracts}
