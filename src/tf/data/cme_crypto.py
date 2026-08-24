"""CME cryptocurrency futures metadata, session regimes, and roll schedules.

Contract specifications here are transcribed from CME product pages. They are
descriptive metadata, not market data: no prices ship with this package.

Sizes and listing dates are stated as of August 2026 and should be re-checked
against the exchange before being relied on, since the crypto lineup is still
expanding.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import pandas as pd

#: Date CME moved its crypto futures and options to near-continuous trading,
#: seven days a week, apart from a short weekly maintenance window. Before this
#: the products traded a roughly five-day Globex week, so a backtest that spans
#: the date spans two session regimes.
CME_CRYPTO_247_START = pd.Timestamp("2026-05-29")

#: Session regime applied to every CME crypto contract.
CME_CRYPTO_SESSION_REGIMES: Mapping[str, str] = {
    "1900-01-01": "weekday",
    CME_CRYPTO_247_START.strftime("%Y-%m-%d"): "daily",
}

#: Month codes used by CME futures, in calendar order.
MONTH_CODES = {
    1: "F", 2: "G", 3: "H", 4: "J", 5: "K", 6: "M",
    7: "N", 8: "Q", 9: "U", 10: "V", 11: "X", 12: "Z",
}

#: Quarterly cycle used by the standard CME crypto contracts.
QUARTERLY_MONTHS = (3, 6, 9, 12)


@dataclass(frozen=True)
class CmeCryptoContract:
    """Specification for one CME cryptocurrency futures contract."""

    symbol: str
    name: str
    contract_size: float
    unit: str
    tick_size: float
    listing_date: str
    reference_ticker: str

    @property
    def point_value(self) -> float:
        """Monetary value of a one-unit move in the underlying index."""

        return self.contract_size

    def as_universe_entry(self, *, data_source: str = "yahoo") -> dict:
        """Return a universe record consumable by :class:`ContractMetadata`."""

        return {
            "symbol": self.symbol,
            "sector": "Crypto",
            "point_value": self.point_value,
            "contract_step": 1.0,
            "tick_size": self.tick_size,
            "contract_size": self.contract_size,
            "listing_date": self.listing_date,
            "calendar": "CRYPTO_DAILY",
            "venue": "CME",
            "data_source": data_source,
            "data_symbol": self.reference_ticker,
            "description": self.name,
            "session_regime_changes": dict(CME_CRYPTO_SESSION_REGIMES),
        }


#: The four contracts with enough history and liquidity to form the v0.1
#: reference universe. ADA, LINK, XLM, AVAX, and SUI are listed but must earn
#: their place through the point-in-time eligibility rule rather than by being
#: hard-coded here.
CME_CRYPTO_CONTRACTS: Mapping[str, CmeCryptoContract] = {
    "BTC": CmeCryptoContract(
        symbol="BTC",
        name="CME Bitcoin Futures",
        contract_size=5.0,
        unit="BTC",
        tick_size=5.0,
        listing_date="2017-12-18",
        reference_ticker="BTC-USD",
    ),
    "ETH": CmeCryptoContract(
        symbol="ETH",
        name="CME Ether Futures",
        contract_size=50.0,
        unit="ETH",
        tick_size=0.05,
        listing_date="2021-02-08",
        reference_ticker="ETH-USD",
    ),
    "SOL": CmeCryptoContract(
        symbol="SOL",
        name="CME Solana Futures",
        contract_size=500.0,
        unit="SOL",
        tick_size=0.05,
        listing_date="2025-03-17",
        reference_ticker="SOL-USD",
    ),
    "XRP": CmeCryptoContract(
        symbol="XRP",
        name="CME XRP Futures",
        contract_size=50_000.0,
        unit="XRP",
        tick_size=0.0001,
        listing_date="2025-05-19",
        reference_ticker="XRP-USD",
    ),
}

#: Reference universe for the v0.1 crypto presets.
REFERENCE_UNIVERSE = ("BTC", "ETH", "SOL", "XRP")


def session_calendar_for(timestamp: pd.Timestamp | str) -> str:
    """Return the CME crypto session calendar in force on ``timestamp``."""

    ts = pd.Timestamp(timestamp).normalize()
    return "daily" if ts >= CME_CRYPTO_247_START else "weekday"


def is_session(timestamp: pd.Timestamp | str) -> bool:
    """Return whether CME crypto traded on ``timestamp``."""

    ts = pd.Timestamp(timestamp).normalize()
    if session_calendar_for(ts) == "daily":
        return True
    return bool(ts.dayofweek < 5)


def next_session(timestamp: pd.Timestamp | str) -> pd.Timestamp:
    """Return the first CME crypto session on or after ``timestamp``.

    A signal generated on a Saturday before the 2026 regime change cannot be
    executed until Monday. Reporting that lag is the point: it is a real cost of
    trading a 7-day underlying on a 5-day venue.
    """

    ts = pd.Timestamp(timestamp).normalize()
    for _ in range(8):
        if is_session(ts):
            return ts
        ts = ts + pd.Timedelta(days=1)
    raise RuntimeError("Could not locate a CME crypto session within eight days")


def fill_lag_days(timestamp: pd.Timestamp | str) -> int:
    """Return how many days a signal at ``timestamp`` waits for a session."""

    ts = pd.Timestamp(timestamp).normalize()
    return int((next_session(ts) - ts).days)


def build_universe(
    symbols: Sequence[str] = REFERENCE_UNIVERSE,
    *,
    data_source: str = "yahoo",
) -> list[dict]:
    """Return universe records for the requested CME crypto contracts."""

    unknown = [s for s in symbols if s.upper() not in CME_CRYPTO_CONTRACTS]
    if unknown:
        raise KeyError(
            f"No CME crypto contract specification for: {', '.join(sorted(unknown))}. "
            f"Available: {', '.join(sorted(CME_CRYPTO_CONTRACTS))}"
        )
    return [
        CME_CRYPTO_CONTRACTS[s.upper()].as_universe_entry(data_source=data_source)
        for s in symbols
    ]


def quarterly_roll_schedule(
    symbol: str,
    start: pd.Timestamp | str,
    end: pd.Timestamp | str,
    *,
    roll_days_before: int = 5,
) -> list[tuple[pd.Timestamp, str]]:
    """Build a quarterly roll schedule for ``symbol``.

    Returns ``(segment_start, contract_code)`` pairs in the shape
    :func:`tf.data.continuous.build_continuous_series` expects, so each pair
    names the contract held *from* that date. CME crypto futures expire on the
    last Friday of the contract month; the schedule moves into the following
    contract ``roll_days_before`` days ahead of each expiry.
    """

    start_ts = pd.Timestamp(start).normalize()
    end_ts = pd.Timestamp(end).normalize()
    if end_ts < start_ts:
        raise ValueError("end must not precede start")
    if roll_days_before < 0:
        raise ValueError("roll_days_before must not be negative")

    # Quarterly expiries spanning the window with one cycle of padding on
    # each side, so the first and last segments have a contract to name.
    expiries: list[tuple[pd.Timestamp, str]] = []
    for year in range(start_ts.year - 1, end_ts.year + 2):
        for month in QUARTERLY_MONTHS:
            expiry = last_friday(year, month)
            code = f"{symbol}{MONTH_CODES[month]}{str(year)[-2:]}"
            expiries.append((expiry, code))
    expiries.sort(key=lambda item: item[0])

    schedule: list[tuple[pd.Timestamp, str]] = []
    for index, (expiry, _code) in enumerate(expiries[:-1]):
        # Five days before this expiry the position moves into the next
        # contract, so the segment starting here is named for that one.
        roll_on = expiry - pd.Timedelta(days=roll_days_before)
        _next_expiry, next_code = expiries[index + 1]
        if start_ts <= roll_on <= end_ts:
            schedule.append((roll_on, next_code))

    # Anchor the first segment at the backtest start with whichever contract is
    # front at that moment, so no history is dropped.
    front_code = _front_contract(expiries, start_ts, roll_days_before)
    if not schedule or schedule[0][0] > start_ts:
        schedule.insert(0, (start_ts, front_code))
    return schedule


def _front_contract(
    expiries: Sequence[tuple[pd.Timestamp, str]],
    timestamp: pd.Timestamp,
    roll_days_before: int,
) -> str:
    """Return the contract held on ``timestamp`` given the roll convention."""

    for expiry, code in expiries:
        if timestamp < expiry - pd.Timedelta(days=roll_days_before):
            return code
    return expiries[-1][1]


def last_friday(year: int, month: int) -> pd.Timestamp:
    """Return the last Friday of ``month``."""

    last_day = pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(1)
    offset = (last_day.dayofweek - 4) % 7
    return (last_day - pd.Timedelta(days=offset)).normalize()
