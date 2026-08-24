"""Cryptocurrency reference-price schema, validation, and universe helpers.

This module defines the shape of a crypto price series and how to build a
universe from it. It deliberately ships no market data and no bundled vendor
credentials: sourcing is constrained by licensing, not by what is reachable.
See ``docs/CRYPTO_DATA.md`` for what each layer may and may not use.
"""

from __future__ import annotations

import logging
from typing import Iterable, Mapping, Sequence

import pandas as pd

from .calendar import TradingCalendar
from .cme_crypto import CME_CRYPTO_CONTRACTS, REFERENCE_UNIVERSE

logger = logging.getLogger(__name__)

#: Columns a raw crypto reference-price file may carry. Only the close is used
#: downstream; the rest are accepted so a file need not be pre-trimmed.
REFERENCE_PRICE_COLUMNS = ("date", "close", "volume_usd", "source", "as_of")

#: Yahoo tickers for the reference universe. These are spot pairs, not futures,
#: and are used as the signal reference series.
SPOT_TICKERS: Mapping[str, str] = {
    symbol: contract.reference_ticker
    for symbol, contract in CME_CRYPTO_CONTRACTS.items()
}


def spot_universe(
    symbols: Sequence[str] = REFERENCE_UNIVERSE,
    *,
    point_value: float = 1.0,
    contract_step: float = 1e-4,
) -> list[dict]:
    """Return a universe of crypto spot instruments.

    Spot is sized in fractional units rather than contracts, so ``point_value``
    is 1 and ``contract_step`` is small. Use this for signal research; use
    :func:`tf.data.cme_crypto.build_universe` when the question is what a CME
    futures implementation would actually have returned.
    """

    entries = []
    for symbol in symbols:
        key = symbol.upper()
        contract = CME_CRYPTO_CONTRACTS.get(key)
        if contract is None:
            raise KeyError(
                f"No reference ticker known for {symbol!r}. "
                f"Available: {', '.join(sorted(CME_CRYPTO_CONTRACTS))}"
            )
        entries.append(
            {
                "symbol": key,
                "sector": "Crypto",
                "point_value": point_value,
                "contract_step": contract_step,
                "calendar": "CRYPTO_DAILY",
                "venue": "SPOT",
                "data_source": "yahoo",
                "data_symbol": contract.reference_ticker,
                "listing_date": contract.listing_date,
                "description": f"{key} spot reference price",
            }
        )
    return entries


def crypto_calendar() -> TradingCalendar:
    """Return the 7-day calendar crypto reference prices are aligned to."""

    return TradingCalendar.from_name("CRYPTO_DAILY")


def validate_crypto_prices(
    prices: pd.DataFrame,
    *,
    max_gap_days: int = 1,
    min_price: float = 0.0,
) -> None:
    """Check a crypto price frame for the failures that matter for trend work.

    Raises rather than warns. A gap or a non-positive price silently forward
    filled becomes a fabricated flat return, which a trend system reads as a
    genuine absence of trend.
    """

    if not isinstance(prices.index, pd.DatetimeIndex):
        raise TypeError("Crypto prices must be indexed by a DatetimeIndex")
    if prices.empty:
        raise ValueError("Crypto price frame is empty")
    if not prices.index.is_monotonic_increasing:
        raise ValueError("Crypto price index must be sorted")
    if prices.index.has_duplicates:
        duplicates = prices.index[prices.index.duplicated()]
        raise ValueError(f"Duplicate dates in crypto prices: {list(duplicates[:5])}")

    gaps = prices.index.to_series().diff().dropna()
    oversized = gaps[gaps > pd.Timedelta(days=max_gap_days)]
    if not oversized.empty:
        first = oversized.index[0]
        raise ValueError(
            f"Crypto prices have a {oversized.iloc[0].days}-day gap at "
            f"{first.date()}; crypto trades every calendar day, so a gap means "
            "missing data rather than a market holiday"
        )

    for column in prices.columns:
        series = prices[column].dropna()
        if series.empty:
            raise ValueError(f"Column {column!r} has no observations")
        if (series <= min_price).any():
            bad = series[series <= min_price]
            raise ValueError(
                f"Column {column!r} has {len(bad)} non-positive prices, "
                f"first at {bad.index[0].date()}"
            )


def data_span(prices: pd.DataFrame) -> pd.DataFrame:
    """Return the first and last observation and the count for each instrument.

    Multi-asset crypto history is short and wildly uneven, so any honest report
    states per instrument when its data actually begins.
    """

    rows = []
    for column in prices.columns:
        series = prices[column].dropna()
        rows.append(
            {
                "symbol": column,
                "first_observation": series.index[0] if not series.empty else pd.NaT,
                "last_observation": series.index[-1] if not series.empty else pd.NaT,
                "observations": int(series.size),
            }
        )
    return pd.DataFrame(rows).set_index("symbol")


def local_csv_universe(
    paths: Mapping[str, str],
    *,
    sector: str = "Crypto",
    point_value: float = 1.0,
    contract_step: float = 1e-4,
) -> list[dict]:
    """Build a universe from local CSV files, one file per instrument.

    ``paths`` maps a symbol to a file path. The loader resolves a price column
    case-insensitively from settle, close, adj_close, price, or last.
    """

    return [
        {
            "symbol": symbol.upper(),
            "sector": sector,
            "point_value": point_value,
            "contract_step": contract_step,
            "calendar": "CRYPTO_DAILY",
            "data_source": "csv",
            "data_symbol": str(path),
        }
        for symbol, path in paths.items()
    ]


def describe_sources(symbols: Iterable[str] = REFERENCE_UNIVERSE) -> pd.DataFrame:
    """Return the reference ticker and listing date for each symbol."""

    rows = []
    for symbol in symbols:
        contract = CME_CRYPTO_CONTRACTS[symbol.upper()]
        rows.append(
            {
                "symbol": contract.symbol,
                "reference_ticker": contract.reference_ticker,
                "cme_listing_date": contract.listing_date,
                "contract_size": contract.contract_size,
                "unit": contract.unit,
            }
        )
    return pd.DataFrame(rows).set_index("symbol")
