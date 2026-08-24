"""Market data ingestion helpers.

The original starter project relied exclusively on synthetic data to keep the
early backtests deterministic.  This module keeps that capability while adding
support for fetching real-world time-series from free data vendors such as
Yahoo! Finance.  The loader attempts to pull live data when possible and falls
back to synthetic simulations for offline or air-gapped environments.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd

from .calendar import TradingCalendar
from .metadata import ContractMetadata, UniverseDefinition
from .validators import validate_price_data
from .synthetic import generate_synthetic_prices

logger = logging.getLogger(__name__)


def load_prices_or_generate(
    universe: Sequence[Mapping[str, object] | ContractMetadata],
    start: str | None = None,
    end: str | None = None,
    *,
    seed: int = 42,
    prefer: str = "auto",
    calendar: TradingCalendar | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    strict: bool = False,
) -> pd.DataFrame:
    """Return daily close prices in wide format for the provided universe.

    Parameters
    ----------
    universe:
        Iterable of contract dictionaries or :class:`ContractMetadata` objects.
    start, end:
        Inclusive ISO date strings bounding the requested history.  Can be
        provided positionally or via keyword arguments ``start``/``end``.
    start_date, end_date:
        Backwards compatible aliases for ``start``/``end``.
    seed:
        Random seed used when synthetic prices are generated.
    prefer:
        ``"auto"`` (default) attempts the configured vendor first and then
        falls back to synthetic data.  ``"yahoo"`` will force Yahoo! Finance
        lookups while ``"synthetic"`` skips vendor queries entirely.
    calendar:
        Optional trading calendar for alignment and validation.  When omitted a
        simple weekday calendar is used.
    strict:
        When ``True`` the synthetic fallback is disabled and any vendor or
        validation failure is raised.  Use it whenever a result depends on the
        data actually being real.
    """

    resolved_start = start or start_date
    resolved_end = end or end_date

    if resolved_start is None or resolved_end is None:
        raise TypeError(
            "load_prices_or_generate() requires 'start'/'end' or 'start_date'/'end_date'"
        )

    if start and start_date and start != start_date:
        raise ValueError("Conflicting start dates provided")
    if end and end_date and end != end_date:
        raise ValueError("Conflicting end dates provided")

    universe_meta = UniverseDefinition.from_payload(universe)
    calendar = calendar or TradingCalendar()

    if prefer not in {"auto", "yahoo", "synthetic"}:
        raise ValueError("Unsupported data preference: %s" % prefer)

    if prefer != "synthetic":
        try:
            prices = _load_from_vendor(universe_meta, resolved_start, resolved_end)
        except Exception as exc:  # pragma: no cover - network failure path
            if strict:
                raise
            logger.warning("Vendor data request failed (%s). Using synthetic.", exc)
        else:
            if not prices.empty:
                # Validation errors are raised, never swallowed: substituting
                # synthetic data for a real series that failed a data-quality
                # check would silently turn a research result into fiction.
                calendar.validate(prices)
                validate_price_data(prices, min_price=0.0, max_consecutive_missing=5)
                aligned = calendar.align(prices, resolved_start, resolved_end)
                validate_price_data(aligned, min_price=0.0, max_consecutive_missing=5)
                return aligned
            if strict:
                raise ValueError(
                    "No vendor data returned for "
                    + ", ".join(c.symbol for c in universe_meta.contracts)
                )
            logger.info(
                "No vendor data returned for %s – falling back to synthetic",
                ", ".join(c.symbol for c in universe_meta.contracts),
            )

    if strict and prefer != "synthetic":
        raise ValueError("strict=True forbids the synthetic price fallback")

    symbols = [c.symbol for c in universe_meta.contracts]
    prices = generate_synthetic_prices(
        symbols,
        resolved_start,
        resolved_end,
        seed=seed,
        freq="D" if calendar.freq == "daily" else "C",
    )
    return calendar.align(prices, resolved_start, resolved_end)


def _load_from_vendor(
    universe: UniverseDefinition, start: str, end: str
) -> pd.DataFrame:
    """Dispatch to vendor specific fetchers and combine the results."""

    grouped: dict[str, list[ContractMetadata]] = defaultdict(list)
    for contract in universe.contracts:
        grouped[contract.data_source.lower()].append(contract)

    frames: list[pd.DataFrame] = []
    for vendor, contracts in grouped.items():
        if vendor == "yahoo":
            frames.append(_fetch_yahoo_prices(contracts, start, end))
        elif vendor in {"csv", "parquet"}:
            frames.append(_load_local_prices(vendor, contracts, start, end))
        else:
            logger.warning("Unsupported data vendor '%s' – skipping", vendor)

    if not frames:
        return pd.DataFrame()

    data = pd.concat(frames, axis=1)
    data = data.loc[:, ~data.columns.duplicated()]
    return data.sort_index()


def _fetch_yahoo_prices(
    contracts: Sequence[ContractMetadata], start: str, end: str
) -> pd.DataFrame:
    """Fetch daily closes for the provided contracts using Yahoo! Finance."""

    try:
        import yfinance as yf  # type: ignore
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError("yfinance is required for Yahoo data downloads") from exc

    ticker_map = {c.vendor_symbol: c.symbol for c in contracts}
    tickers = list(ticker_map.keys())
    if not tickers:
        return pd.DataFrame()

    logger.info(
        "Fetching Yahoo! Finance data for %s", ", ".join(sorted(ticker_map.values()))
    )

    data = yf.download(
        tickers=tickers if len(tickers) > 1 else tickers[0],
        start=start,
        end=end,
        auto_adjust=False,
        progress=False,
        threads=False,
    )

    if data is None or len(data) == 0:
        return pd.DataFrame(columns=ticker_map.values())

    closes = _extract_close_prices(data)
    if closes.shape[1] == 1 and len(tickers) == 1:
        closes.columns = [tickers[0]]
    closes = closes.rename(columns=ticker_map)

    expected_cols = [c.symbol for c in contracts]
    return closes.reindex(columns=expected_cols)


def _extract_close_prices(data: pd.DataFrame) -> pd.DataFrame:
    """Handle the different return shapes emitted by :func:`yfinance.download`."""

    if isinstance(data.columns, pd.MultiIndex):
        for candidate in ("Adj Close", "Close"):
            if candidate in data.columns.levels[0]:
                closes = data[candidate]
                break
        else:  # pragma: no cover - unexpected vendor schema
            raise KeyError("Unable to locate Close prices in Yahoo! data")
        if isinstance(closes, pd.Series):
            closes = closes.to_frame()
        return closes

    # Single ticker path - standard dataframe with OHLC columns
    if "Adj Close" in data.columns:
        closes = data[["Adj Close"]]
    elif "Close" in data.columns:
        closes = data[["Close"]]
    else:  # pragma: no cover - unexpected vendor schema
        raise KeyError("Yahoo! response missing 'Close' column")

    return closes


def _load_local_prices(
    vendor: str, contracts: Sequence[ContractMetadata], start: str, end: str
) -> pd.DataFrame:
    frames: list[pd.Series] = []
    for contract in contracts:
        if not contract.data_symbol:
            raise ValueError(
                f"Contract '{contract.symbol}' requires 'data_symbol' when using {vendor} source"
            )
        path = Path(contract.data_symbol).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Data file '{path}' for contract '{contract.symbol}' not found")
        series = _read_price_series(path, vendor)
        frames.append(series.rename(contract.symbol).loc[start:end])
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, axis=1).sort_index()


_COLUMN_CANDIDATES = ("settle", "close", "adj_close", "price", "last")


def _read_price_series(path: Path, vendor: str) -> pd.Series:
    if vendor == "csv":
        frame = pd.read_csv(path)
    elif vendor == "parquet":
        frame = pd.read_parquet(path)
    else:  # pragma: no cover - defensive
        raise ValueError(f"Unsupported local vendor '{vendor}'")

    if "date" in frame.columns:
        frame["date"] = pd.to_datetime(frame["date"])
        frame = frame.set_index("date").sort_index()
    elif frame.index.name is None or not isinstance(frame.index, pd.DatetimeIndex):
        raise ValueError(
            f"Data file '{path}' must contain a 'date' column or be indexed by DatetimeIndex"
        )

    if isinstance(frame, pd.Series):
        return frame.astype(float)

    lowered = {col.lower(): col for col in frame.columns}
    for candidate in _COLUMN_CANDIDATES:
        if candidate in lowered:
            return frame[lowered[candidate]].astype(float)

    numeric = frame.select_dtypes(include="number")
    if numeric.shape[1] == 1:
        return numeric.iloc[:, 0].astype(float)

    raise ValueError(
        f"Could not determine price column for '{path}'. Expected one of {_COLUMN_CANDIDATES}"
    )
