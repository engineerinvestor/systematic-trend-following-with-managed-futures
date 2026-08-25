# Crypto Extension Specification (`tf-trend[crypto]`)

Maintained by Engineer Investor ([@egr_investor](https://x.com/egr_investor)).

Status: implemented in v0.1. This document specifies a cryptocurrency extension
to the existing `tf-trend` research engine. It pairs with `SPEC.md` (core
architecture) and reuses the existing data, signal, risk, engine, and research
layers wherever possible. Section 13 records where the implementation departed
from this specification and why.

## 1. Purpose

Provide a reproducible, point-in-time, cost-aware implementation of published
institutional trend-following methodology applied to cryptocurrency markets: the
academic time-series momentum (TSMOM) strategy of Moskowitz, Ooi, and Pedersen, and a
multi-system bottom-up ensemble of the kind described in ReSolve's CTA replication
research. The package takes no position on whether crypto trend following works; it is
built to measure that question honestly, including the possibility that the answer is no.

### Goals

- Implement the public MOP (2012) TSMOM methodology adapted to a 7-day, 365-day-year
  trading calendar.
- Implement a bottom-up multi-system ensemble (momentum, price vs. moving average,
  dual moving average, breakout) across a ladder of horizons.
- Support long/short and long/flat variants of every strategy and report both.
- Enforce point-in-time discipline everywhere: signal lag, universe eligibility,
  and data availability.
- Ship a falsification suite that stress-tests every backtest against naive
  benchmarks, cost multiples, and signal perturbations by default.

### Non-goals for v0.1

- No top-down replication. ReSolve's top-down approach regresses against a target
  index of live CTA returns. No diversified crypto CTA index with meaningful history
  exists to serve as that target; current US-listed crypto trend products are
  long/flat single-asset or two-asset strategies (Global X BTRN tracks the CoinDesk
  Bitcoin Trend Indicator; Bitwise's BTOP was liquidated in May 2026). A generic
  `TopDownReplicator` may land later as an optional component with a user-supplied
  target series.
- No RSI, MACD, machine learning, sentiment, or on-chain features. The value of v0.1
  is that every line of strategy code traces to a published paper.
- No live order execution, exchange API keys, or order management. Research only.
- No perpetual futures execution modeling (funding-rate accrual, liquidation
  mechanics). The instrument abstraction reserves a slot for a later
  `data.perpetuals` adapter.

## 2. Evidence base

Every strategy component in this extension traces to one of the following sources.

| Claim | Source |
|---|---|
| An instrument's own 12-month past return predicts its next-month return across 58 futures/forwards; positions scaled inversely to lagged volatility | Moskowitz, Ooi, Pedersen (2012), "Time Series Momentum," *Journal of Financial Economics* 104(2) |
| A 1/3/12-month TSMOM ensemble with vol targeting explains much of managed-futures index behavior | Hurst, Ooi, Pedersen (2013), "Demystifying Managed Futures," *Journal of Investment Management* 11(3) |
| Trend following earned positive returns in most decades back to 1880 | Hurst, Ooi, Pedersen (2017), "A Century of Evidence on Trend-Following Investing," *Journal of Portfolio Management* 44(1) |
| A meaningful share of published TSMOM performance comes from volatility scaling rather than the directional signal | Kim, Tse, Wald (2016), "Time series momentum and volatility scaling," *Journal of Financial Markets* 30 |
| Time-series momentum is present in cryptocurrency returns | Liu, Tsyvinski (2021), "Risks and Returns of Cryptocurrency," *Review of Financial Studies* 34(6) |
| Momentum is one of a small number of priced cryptocurrency factors | Liu, Tsyvinski, Wu (2022), "Common Risk Factors in Cryptocurrency," *Journal of Finance* 77(2) |
| Crypto trend portfolios show peak risk-adjusted returns near 10 to 15 coins before costs and liquidity dominate | Man Group, "In Crypto We Trend" (man.com/insights/in-crypto-we-trend) |
| Bottom-up CTA replication with roughly thirteen lookbacks across several trend-system families, weighted by Ridge regression | ReSolve Asset Management, "How to Replicate Trend Following Managed Futures" (investresolve.com) |
| Long-sample evidence on crypto trend rules | Zarattini, Pagani, Wilcox (arXiv 2009.12155), "A Decade of Evidence of Trend Following Investing in Cryptocurrencies" |

Rules for extending this table: a component with no published source does not ship in a
preset. Named products (funds, ETFs, indexes) are described only from issuer documents
or filings, and those descriptions carry an as-of date.

## 3. Calendar and annualization

This is the largest change to the core engine and a precondition for everything else.

The existing engine is bar-count based: lookbacks like `[63, 126, 252]` mean trading
days on a weekday calendar (`TradingCalendar.sessions` uses `pd.bdate_range`), and
volatility annualizes with `periods_per_year=252`. Crypto reference prices exist every
calendar day. Pretending crypto has 252 trading days misaligns every horizon and
understates annualized volatility by roughly `sqrt(365/252) - 1` (about 20%).

### 3.1 Required changes

- `TradingCalendar` gains a `freq` mode: `"weekday"` (current behavior) and
  `"daily"` (all calendar days). Crypto instruments declare `calendar: "CRYPTO_DAILY"`
  in universe metadata, which maps to the daily mode.
- `ewma_vol` and `rolling_volatility` already accept `periods_per_year`; crypto
  configs must set 365. `eval/metrics.py` annualization helpers gain the same
  parameter, threaded from config, replacing any hard-coded 252.
- Signal horizons in crypto configs are written as calendar durations (`30D`, `90D`,
  `365D`) and resolved to bar counts against the instrument's calendar at config-load
  time. On the daily calendar, `30D` resolves to 30 bars. This keeps the core signal
  functions (which take integer lookbacks) unchanged.

### 3.2 The two-calendar problem

CME crypto futures traded a roughly 5-day CME Globex week from listing until May 29,
2026, when CME moved its crypto products to around-the-clock trading, seven days a
week, with brief maintenance windows. Spot and reference-rate series are 7-day for
their entire history. A single backtest therefore spans a calendar regime change on
the execution venue.

The design rule: **signals are computed on a 7-day reference-price series; execution
happens on the venue calendar through an adapter.** Concretely:

- The signal layer consumes daily reference prices (e.g., exchange reference rates or
  a composite spot series) on the daily calendar, for the full history.
- The execution layer holds a per-instrument session calendar. An order generated
  from a Saturday signal on pre-2026 CME data fills at the next available session
  (Sunday evening/Monday open), and the fill-lag is recorded so its cost is measurable.
- The venue calendar regime change is data (a dated property of the instrument
  metadata), never a hard-coded constant.

## 4. Reference strategies

Presets are named for the methodology they implement. None is named for a fund
manager or a ticker, because none claims to replicate any specific product.

### 4.1 `mop2012_tsmom`

The canonical benchmark. For instrument *i* with daily log returns
$r_{i,t} = \ln(P_{i,t}/P_{i,t-1})$:

- Signal per horizon *h*: $s_{i,h,t} = \mathrm{sign}\left(\sum_{\tau=t-h}^{t-1} r_{i,\tau}\right)$
  with $h \in \{365\mathrm{D}\}$ for the single-horizon variant.
- Volatility: exponentially weighted daily variance with a 60-day center of mass
  (the MOP estimator), lagged one day, annualized by $\sqrt{365}$. In the existing
  `ewma_vol(lam=...)` parameterization, a 60-day center of mass corresponds to
  `lam = 60/61` (approximately 0.9836), since center of mass $= \lambda/(1-\lambda)$.
- Position: $\tilde w_{i,t} = s_{i,t} \cdot \sigma_{\text{inst}} / \hat\sigma_{i,t-1}$
  with $\sigma_{\text{inst}} = 0.40$.
- The combined portfolio is then scaled to `target_portfolio_vol` (default 0.10).

The 40% figure is MOP's per-instrument normalization convention. It is not a
recommendation to run a crypto portfolio at 40% volatility, and the portfolio-level
scaling step is mandatory, together with the gross-exposure limit in Section 6.

**Replication trap:** the existing `timeseries_momentum` defaults to
`skip_last_n=20`, the skip-month convention from cross-sectional momentum. MOP TSMOM
uses no skip. Crypto presets set `skip_last_n: 0` explicitly, and the preset loader
refuses to run a config labeled `mop2012_tsmom` with a nonzero skip.

### 4.2 `tsmom_1_3_12`

The Hurst, Ooi, Pedersen ensemble: equal-weighted sign signals at 30D, 90D, and 365D
horizons, same volatility machinery as `mop2012_tsmom`. This is the recommended
default TSMOM preset.

### 4.3 `bottom_up_multisystem`

A ReSolve-style bottom-up ensemble over four signal families, reusing the existing
signal modules:

| Family | Existing module | Definition |
|---|---|---|
| Total-return momentum | `signals/momentum.py` | Sign of trailing total return over horizon *h* |
| Price vs. moving average | `signals/moving_average.py` | Sign of price minus its *h*-day moving average |
| Dual moving average | `signals/moving_average.py` | Sign of fast MA minus slow MA (fast = *h*/4, floor 2) |
| Breakout | `signals/breakout.py` | Donchian: +1 above the *h*-day high band, -1 below the low band, else hold |

Default horizon ladder: 5, 10, 15, 20, 30, 40, 60, 90, 120, 150, 180, 220, 260 days.
The thirteen-horizon multi-family design follows ReSolve's published replication
research; this specific day ladder is this package's default, chosen to span one week
to one year with denser coverage at the short end.

Family and horizon weights are equal in v0.1. ReSolve fits Ridge-regression weights
against a target index; with no crypto target index and a short joint history,
estimated weights would be noise. ReSolve's robustness work also reports that the
individual family choice mattered little in their tests, which argues for
diversifying simple systems rather than tuning one. Combined signal:
$S_{i,t} = \frac{1}{|F||H|}\sum_{f \in F}\sum_{h \in H} s^{f}_{i,h,t}$, then the same
volatility normalization and portfolio scaling as Section 4.1, with a 40-day EWMA
volatility estimate (`lam = 40/41`) to match the replication literature's convention.

### 4.4 `btc_long_flat`

A deliberately simple comparison strategy: 200-day price-vs-moving-average on BTC
alone, long or flat, vol-targeted. This is the shape of most live retail crypto trend
products and must appear in every falsification report as a benchmark.

### 4.5 Long/short and long/flat

Every preset accepts `direction: long_short` (signals in {-1, +1}) or
`direction: long_flat` (negative signals map to 0). The classical managed-futures
evidence is long/short; the case for long/flat in crypto rests on positive
unconditional drift, borrow cost, and squeeze risk on the short side. The package
publishes both variants side by side in reports and does not privilege either.

## 5. Universe and point-in-time eligibility

### 5.1 Reference universe

The v0.1 institutional reference universe is **BTC, ETH, SOL, XRP**, implemented as
CME futures with a spot/reference-rate signal series. CME listed SOL futures in March
2025 and XRP futures in May 2025; ADA, LINK, and XLM followed in early 2026 and AVAX
and SUI in May 2026. The newer contracts enter only by passing the eligibility rule
below on point-in-time data, never by editorial decision.

### 5.2 Eligibility rule schema

```yaml
universe:
  eligibility:
    min_history: 365D            # observed reference-price history
    min_adv_usd_30d: null        # calibrate from data; null disables
    min_open_interest_usd_30d: null
    max_stale_days: 1            # consecutive missing/stale sessions
    evaluation_frequency: monthly
    entry_lag: 30D               # rule must pass for this long before entry
```

Eligibility is evaluated with data available at the evaluation date. Running today's
top-ten coins backward through history is survivorship bias and the engine must make
it impossible to do by accident: universe membership is a time-indexed mask, and the
backtester consumes the mask, never a static symbol list, when eligibility is enabled.

### 5.3 History honesty

Multi-asset crypto futures history is short. Before 2025 a CME-implemented portfolio
holds only BTC (Dec 2017) and ETH (Feb 2021); spot history adds a few years more.
Every report states, per instrument, the first date it entered the universe, and the
falsification suite includes a data-span panel (Section 8). Walk-forward folds will
be few; confidence intervals must reflect that rather than hide it.

## 6. Risk and sizing

Reuse `portfolio/sizing.py` (`volatility_target_positions`, `_scale_to_gross_limit`,
`_apply_sector_caps`) with crypto-specific configuration:

- `target_portfolio_vol`: 0.10 default (0.12 acceptable; anything above 0.15
  triggers a config warning).
- `gross_exposure_limit`: required, default 3.0. The 40% instrument convention plus a
  low-vol regime produces large gross exposure without a cap.
- Per-asset cap: new `max_asset_weight` (default 0.50 of gross) added alongside the
  existing sector caps. A four-asset universe with one dominant trend will otherwise
  concentrate into a single coin.
- Correlation: crypto pairwise correlations are high and regime-dependent. v0.1 uses
  the existing proportional risk allocator with the caps above. A covariance-aware
  allocator is a v0.2 candidate, with the correlation matrix reported either way.

## 7. Data layer

### 7.1 New modules

| Module | Responsibility |
|---|---|
| `src/tf/data/crypto.py` | Canonical daily reference-price schema (date, symbol, close, volume_usd, source, as_of); validation; 7-day calendar alignment |
| `src/tf/data/cme_crypto.py` | CME contract metadata (contract size, tick, listing date, session calendar with the May 2026 regime date), expiry schedule, roll schedule feeding the existing `data/continuous.py` builders |
| `src/tf/costs/crypto.py` | Commission per contract, half-spread by instrument and era, participation impact reusing `engine/execution.participation_slippage`, roll cost |

The existing `data/ingest.py` vendor interface (local CSV, Yahoo, synthetic) is the
extension point; crypto sources plug in as vendors and every vendor records the
license basis for its data.

### 7.2 Sourcing and licensing

Sourcing is constrained by licensing, not by what is reachable. Web-scraping an
exchange or redistributing licensed market data in a public repo is not acceptable,
so each data need states its license-clean options:

| Need | License-clean options | Notes |
|---|---|---|
| Daily spot/reference prices | Exchange public market-data APIs under their own terms (user fetches with their own access); user-supplied CSV | The package ships fetch adapters and schema validation. It does not redistribute price data. |
| CME futures daily settles, volume, OI | CME licensed data (user-supplied); CME's public delayed pages are not licensed for redistribution | v0.1 ships the schema, roll logic, and synthetic fixtures; users bring their own CME data. |
| Point-in-time market cap / volume rankings | **No obviously license-clean free source.** CoinMetrics community data is CC BY-NC-SA (non-commercial, share-alike); commercial aggregator APIs restrict redistribution | v0.1 therefore keys eligibility on price/volume history the user already holds, and documents this gap prominently instead of quietly bundling scraped rankings. |
| Perpetual funding rates | Deferred with the perpetuals adapter | |

Repository policy: the repo contains synthetic fixtures and schema documentation
only. No historical market data is committed.

## 8. Falsification suite

`src/tf/eval/falsification.py`, surfaced as `tf crypto falsify <config>`. Built on
the existing `research/walkforward.py` and `research/sensitivity.py`.

### 8.1 The vol-scaling 2x2

Kim, Tse, and Wald show that volatility scaling accounts for a meaningful share of
published TSMOM performance. Every falsification report therefore decomposes the
strategy into four variants, run on identical data and costs:

| Variant | Trend signal | Vol scaling |
|---|---|---|
| Buy and hold | no | no |
| Vol-managed buy and hold | no | yes |
| Raw TSMOM | yes | no |
| Vol-managed TSMOM | yes | yes |

The report states which cell the headline performance comes from.

### 8.2 Benchmark and stress battery

Each run of `falsify` compares the configured strategy against, at minimum:

- BTC buy-and-hold and equal-risk crypto buy-and-hold
- `btc_long_flat` (200-day, Section 4.4)
- Single-horizon 12-month TSMOM
- Signal delayed by 1, 2, and 5 additional days
- Transaction costs at 1x, 2x, and 4x the configured model
- Alternative volatility lookbacks (20D, 40D, 60D, 90D)
- Leave-one-asset-out portfolios
- Randomized-sign signal placebo (distribution over 1,000 draws)

plus long/short vs. long/flat, gross vs. net, long-side vs. short-side attribution,
per-asset attribution, and the data-span panel: per-instrument history start, count of
walk-forward folds, and the fraction of backtest days on which the universe held
fewer than three eligible assets.

### 8.3 Point-in-time enforcement

Backtest configs enforce `signals.lag >= 1`: the engine refuses to run a config
asking for a zero-lag signal. Library signal functions do accept `lag=0` for
research on the signal itself; nothing that produces a backtest number does.

Measured end to end, the engine's delay is **two bars**: a price event on day T
first changes the position at day T+2's fill, at that session's **close** (the
data model is close-only; there are no open prices). One bar comes from the
signal's internal lag and one from the order-submission mechanics. This is one
bar more conservative than the minimum honest implementation, so published
results are, in effect, "delay +1" numbers; the falsification suite's delay
rows stack on top of this baseline.

## 9. Configuration schema

`configs/crypto/tsmom_1_3_12.yaml` (reference example):

```yaml
data:
  root_dir: ./data
  calendar: "CRYPTO_DAILY"
  vendor: local            # user-supplied files; see docs/CRYPTO_DATA.md
  continuous_method: "stitch_returns"

universe:
  assets_file: ./configs/crypto/universe.yaml   # BTC, ETH, SOL, XRP + metadata
  eligibility:
    min_history: 365D
    max_stale_days: 1
    evaluation_frequency: monthly
    entry_lag: 30D

signals:
  preset: tsmom_1_3_12
  horizons: ["30D", "90D", "365D"]
  skip_last_n: 0
  direction: long_short     # or long_flat

risk:
  vol_model: "ewma"
  ewma_center_of_mass: 60D  # resolved to lam = com/(com+1)
  periods_per_year: 365
  instrument_vol_target: 0.40
  target_portfolio_vol: 0.10
  gross_exposure_limit: 3.0
  max_asset_weight: 0.50

execution:
  order_type: "market_next_open"
  venue: cme
  signal_lag: 1D

backtest:
  start: "2018-01-01"
  rebalance: weekly
  results_dir: "./results"
```

## 10. Module map

| Path | Status | Work |
|---|---|---|
| `src/tf/data/calendar.py` | modify | `freq` mode for daily calendars; per-instrument session calendars with dated regime changes |
| `src/tf/data/crypto.py` | new | reference-price schema, validation, vendor adapters |
| `src/tf/data/cme_crypto.py` | new | contract metadata, expiries, roll schedule |
| `src/tf/data/continuous.py` | reuse | continuous series from crypto roll schedules |
| `src/tf/signals/*` | reuse | called with resolved bar counts; `skip_last_n=0` enforced by presets |
| `src/tf/risk/vol.py` | reuse | `periods_per_year=365`, center-of-mass parameterization exposed in config |
| `src/tf/portfolio/sizing.py` | modify | add `max_asset_weight`; accept time-indexed universe mask |
| `src/tf/costs/crypto.py` | new | crypto cost model |
| `src/tf/engine/*` | modify | consume universe mask; fill-lag accounting across venue calendars |
| `src/tf/eval/metrics.py` | modify | thread `periods_per_year` through annualization |
| `src/tf/eval/falsification.py` | new | Section 8 |
| `src/tf/cli.py` | modify | `tf crypto falsify` subcommand |
| `configs/crypto/` | new | presets and universe file |
| `docs/CRYPTO_DATA.md` | new | data schema, sourcing, and licensing guidance |

## 11. v0.1 milestones and acceptance criteria

Milestones, in order:

1. **Calendar and annualization.** Daily-calendar mode; 365 threading through vol and
   metrics. *Accept when:* a synthetic 7-day random-walk series with known daily vol
   reports annualized vol within 1% of `sigma * sqrt(365)`, and all existing
   weekday-calendar tests still pass unchanged.
2. **Crypto data layer.** Schema, validation, local-CSV vendor, synthetic crypto
   fixtures, CME metadata with the session regime change. *Accept when:* fixture data
   for four instruments round-trips ingest, validation, and continuous construction,
   and a Saturday signal on a pre-2026 CME session calendar fills on the next session
   with the lag recorded.
3. **Presets.** `mop2012_tsmom`, `tsmom_1_3_12`, `bottom_up_multisystem`,
   `btc_long_flat`, each in long/short and long/flat. *Accept when:* on a synthetic
   trending series each preset holds the expected sign; on a synthetic mean-reverting
   series TSMOM's gross return is negative; a `mop2012_tsmom` config with
   `skip_last_n != 0` refuses to run.
4. **Eligibility mask.** Point-in-time universe machinery. *Accept when:* an
   instrument whose synthetic history starts mid-backtest enters the portfolio no
   earlier than `min_history + entry_lag` after its first data point, verified from
   position artefacts.
5. **Falsification suite.** Section 8 report. *Accept when:* `tf crypto falsify` on
   the reference config emits the 2x2 table, benchmark battery, attribution, and
   data-span panel in one HTML/markdown bundle, and the randomized-sign placebo's
   mean Sharpe is within noise of zero.
6. **Docs.** `docs/CRYPTO_DATA.md`, README section, worked notebook on fixtures.

All acceptance tests run on synthetic or user-supplied data; CI never fetches market
data.

## 12. Disclaimer

This software implements methodologies described in public academic and practitioner
research. It does not reproduce, and does not claim to reproduce, any proprietary
strategy of AQR, ReSolve Asset Management, Return Stacked ETFs, Man AHL, or any other
manager or product. Nothing in this repository is investment advice. Cryptocurrency
markets carry substantial risk, including total loss.

## 13. Implementation notes

Where building v0.1 contradicted this specification, the specification is wrong
and these notes are right.

### Presets are equal-weighted over the horizons named here

Section 4.3's thirteen-horizon ladder is this package's default, not any
manager's published constant. CTA replication research describes thirteen
lookbacks spanning weeks to over a year with Ridge-fitted weights; the specific
day values and the equal weighting are ours.

### Time-series momentum uses sign signals, not tanh strength

The existing `timeseries_momentum` returned a tanh-squashed magnitude. The
published construction takes the sign, so `transform="sign"` was added and the
presets use it. With sign signals the existing proportional risk budget reduces
to equal weight across active instruments, which is the intended construction.

### Two defects blocked correct implementation and were fixed

`channel_breakout` included the current bar in its own high-low window, so it
could never signal a breakout; `donchian_breakout` was added and the original
is documented as a channel-position oscillator. Position sizing back-filled the
volatility warm-up from data that had not yet arrived, which was lookahead bias
in every backtest the package produced. Both are recorded in `CHANGELOG.md`.

### The placebo is vectorised and gross of execution

Section 8.2 specifies 1,000 randomised-sign draws. A thousand full order-queue
backtests take roughly twenty minutes, so the placebo uses a vectorised
signal-to-NAV path that skips queueing and fills. A placebo tests whether a
signal's direction carried information, not whether it could be executed. The
report labels the figure accordingly.

The placebo flips one sign per instrument, so a universe of *n* instruments has
only 2^n distinct outcomes. With the four-asset reference universe that is 16,
and the reported percentile is correspondingly coarse. The report says so.

### Mixed futures and cryptocurrency universes are unsupported

All instruments share one index and one calendar. A mixed universe would either
forward fill futures across weekends, understating their volatility by roughly
the square root of five sevenths, or discard two sevenths of the crypto
observations. Per-instrument calendars are the prerequisite; until then crypto
universes are crypto-only.

### Costs are indicative, not net

The engine's slippage is a tick count times a single universe-wide tick value,
so it scales with quantity rather than notional. Crypto prices span five orders
of magnitude, so one tick value cannot serve the whole universe and a
plausible-looking configuration ends up nearly frictionless: on a real
BTC/ETH/SOL run turning over roughly 9 times per year, quadrupling every
configured cost moved annual return by 6.5 basis points. `tf.costs.crypto.CryptoCostModel`
states spreads correctly, in basis points of notional per instrument and era,
but the execution layer does not yet consume it. The falsification report
detects the condition and reports it rather than presenting gross figures as
net. Teaching the execution layer about proportional costs is the first item
for v0.2.

### There is no margin model

The engine debits the full notional of every fill from cash, so a strategy is
modelled as an unlevered cash account with implicit zero-cost financing. This is
adequate for comparing strategies on equal terms and inadequate as a
financing-accurate simulation of a futures programme.

### Leading gaps are not data-quality failures

Section 5.3 anticipated short histories but not their interaction with
validation. Missing observations before an instrument's first price mean it did
not exist yet, so `validate_price_data` permits them by default. Gaps inside an
observed history are still rejected: pulling XRP daily candles from a major US
exchange returns a 905-day hole from January 2021 to July 2023, when that venue
suspended XRP trading, and forward filling it would have manufactured two and a
half years of zero returns mid-sample.
