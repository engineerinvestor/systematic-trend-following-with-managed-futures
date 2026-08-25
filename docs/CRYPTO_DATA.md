# Crypto Data: Schema, Sourcing, and Limits

This guide covers what the crypto extension expects from price data, where that
data may legitimately come from, and the modelling limits you should know before
trusting a result. The methodology itself is in `CRYPTO_SPEC.md`.

**No market data ships with this repository, and none should be committed to it.**

## Price schema

The loader is close-only. Each instrument resolves to one price series, indexed
by date, on a 7-day calendar.

A CSV needs a `date` column and one price column, resolved case-insensitively
from `settle`, `close`, `adj_close`, `price`, or `last`. Extra columns are
accepted and discarded.

```csv
date,close,volume_usd
2024-01-01,42280.23,18432000000
2024-01-02,44167.33,26714000000
```

Point the universe at the file:

```python
from tf.data.crypto import local_csv_universe

universe = local_csv_universe({"BTC": "/path/to/btc.csv", "ETH": "/path/to/eth.csv"})
```

Validation is deliberately strict. `validate_crypto_prices` raises on calendar
gaps, duplicate dates, and non-positive prices rather than warning. A gap that
gets forward filled becomes a fabricated flat return, and a trend system reads a
run of flat returns as a genuine absence of trend.

This is not hypothetical. Pulling XRP daily candles from a major US exchange for
2017 to 2025 returns a series with a 905-day hole between 19 January 2021 and 13
July 2023, when that venue suspended XRP trading. Forward filling it would have
manufactured two and a half years of zero returns in the middle of the sample.
Leading gaps are treated differently: missing observations before an
instrument's first price mean it did not exist yet, not that data is missing, so
`validate_price_data(allow_leading_gaps=True)` permits them and
`data.allow_partial_history` handles them in the engine.

Pass `strict=True` to `load_prices_or_generate` whenever a result depends on the
data being real. Without it, a vendor or network failure falls back to synthetic
prices, which is convenient for smoke tests and catastrophic for research.

## Sourcing

Sourcing is constrained by licensing, not by what is reachable. Scraping an
exchange or redistributing licensed market data is not an option, so each layer
has its own answer.

| Need | License-clean options | Notes |
|---|---|---|
| Daily spot reference prices | Exchange public market-data APIs under their own terms, fetched with your own access; user-supplied CSV; Yahoo via the existing vendor path for personal research | The package ships adapters and validation. It does not redistribute price data. |
| CME futures settles, volume, open interest | CME licensed data, user-supplied | The package ships contract specifications, roll logic, and synthetic fixtures. Delayed public pages are not licensed for redistribution. |
| Point-in-time market cap and volume rankings | **No obviously license-clean free source.** Community datasets are typically non-commercial share-alike; commercial aggregator APIs restrict redistribution | This is the real gap. Eligibility therefore keys on price and volume history you already hold, rather than on bundled rankings. |
| Perpetual funding rates | Deferred along with perpetual execution support | |

For a quick real-data run, the Yahoo vendor path already handles crypto spot
because `data_symbol` passes through unchanged:

```yaml
- symbol: BTC
  sector: Crypto
  point_value: 1.0
  contract_step: 0.0001
  data_source: yahoo
  data_symbol: BTC-USD
```

Set `data.prefer: auto` and `data.strict: true` in the config to use it. Check
the vendor's terms for your own use; nothing is cached into the repository.

## The two-calendar problem

Signals are computed on a 7-day reference series. Execution happens on the
venue's calendar.

CME crypto futures traded a roughly five-day week until 29 May 2026, when the
products moved to around-the-clock trading. A backtest spanning that date spans
two session regimes, which `tf.data.cme_crypto` stores as dated metadata rather
than as a constant:

```python
from tf.data.cme_crypto import fill_lag_days, next_session

fill_lag_days("2024-06-01")   # 2: a Saturday signal waits for Monday
fill_lag_days("2026-07-04")   # 0: the same weekend after the change
```

That lag is a real cost of trading a 7-day underlying on a 5-day venue, so it is
reported rather than assumed away.

## Universe eligibility

Running today's leading cryptocurrencies backwards through history is
survivorship bias. Eligibility is a time-indexed mask, evaluated only on data
available at each evaluation date:

```yaml
universe:
  eligibility:
    min_history: 365D          # observations before an instrument may trade
    max_stale_days: 1          # consecutive missing bars that disqualify it
    evaluation_frequency: monthly
    entry_lag: 30D             # the rule must hold this long before entry
```

Set `data.allow_partial_history: true` alongside it. Without that flag the
engine truncates the backtest to the intersection of all histories, so adding
one recently listed coin silently discards years of data from everything else.

Check what you actually held:

```python
from tf.data.eligibility import build_eligibility_mask, eligibility_summary

mask = build_eligibility_mask(prices, rule)
eligibility_summary(mask)
```

Expect this to be humbling. A four-asset crypto universe is usually one asset
for most of its history: CME listed Bitcoin futures in December 2017, Ether in
February 2021, and Solana and XRP only in 2025.

## Modelling limits

Read these before quoting a number from this package.

**No margin model.** The engine debits the full notional of every fill from
cash. Futures are margined instruments, so this is wrong in a specific way: it
treats the strategy as an unlevered cash account with implicit zero-cost
financing. Gross exposure above 1 drives cash negative and the backtest keeps
running. Results are usable for comparing strategies on equal terms; they are
not a financing-accurate simulation.

**Costs are assumptions, not measurements.** The defaults in `tf.costs.crypto`
are plausible starting points, wider for newer contracts and wider again before
2021. Treat the cost stress rows in the falsification report as the real output:
if a result survives only at 1x assumed costs, it is a cost assumption, not a
strategy.

**Configure proportional spreads or costs will barely bite.** Tick-based
slippage scales with quantity rather than notional, which cannot serve a
universe where BTC near $60,000 and XRP near $0.50 differ by five orders of
magnitude: with only the tick model, a real BTC/ETH/SOL run turning over
roughly 9 times per year saw 6.5 basis points of annual return between 1x and
4x costs. The engine therefore also charges `execution.spread_bps`, a
per-instrument half-spread in basis points of notional (the shipped crypto
configs carry values from `tf.costs.crypto`), reads per-instrument tick values
from metadata `tick_size`, and honours `execution.cost_multiplier` as one
honest stress knob. The falsification report still detects and says when the
configured costs are too small to constrain the strategy.

**Mixed futures and crypto universes are unsupported.** All instruments share
one index and one calendar. Aligning a mixed universe to a 7-day calendar
forward fills futures across weekends, adding two zero-return days a week and
understating their volatility by roughly the square root of five sevenths;
aligning to a weekday calendar discards two sevenths of the crypto
observations. Run crypto-only universes until per-instrument calendars exist.

**Signals are not volatility-normalised across instruments.** Within a
single-sector crypto universe this cancels, because the risk budget normalises
by the row sum of absolute signals and direction uses only the sign. It would
not cancel in a mixed universe, which is a second reason not to build one yet.

**History is short.** Multi-asset crypto backtests have few independent
observations and correspondingly few walk-forward folds. The data-span panel in
the falsification report exists so that this is visible rather than implied.

## Reproducing a result

```bash
tf run --config configs/crypto/tsmom_1_3_12.yaml --run-id my-run
tf crypto falsify --config configs/crypto/tsmom_1_3_12.yaml --run-id my-run-falsify
```

The shipped configs use `data.prefer: synthetic` so they run offline out of the
box. Synthetic prices are generated, not real: switch to `auto` with `strict:
true` and your own data before drawing any conclusion.
