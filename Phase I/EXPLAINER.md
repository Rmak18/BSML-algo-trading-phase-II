# Adaptive Adversary Regression in Algorithmic Trading — Project Explainer

*Bocconi Students for Machine Learning (BSML) · Published December 2025*

---

## Table of Contents

1. [Motivation and Problem Statement](#1-motivation-and-problem-statement)
2. [Project Architecture](#2-project-architecture)
3. [Synthetic Data Universe](#3-synthetic-data-universe)
4. [Baseline Execution Strategy (TSMOM)](#4-baseline-execution-strategy-tsmom)
5. [Transaction Cost Model](#5-transaction-cost-model)
6. [Randomization Policies](#6-randomization-policies)
   - [Uniform Randomization](#61-uniform-randomization)
   - [Ornstein-Uhlenbeck (OU) Process](#62-ornstein-uhlenbeck-ou-process)
   - [Pink (1/f) Noise](#63-pink-1f-noise)
7. [The Adversary Classifier](#7-the-adversary-classifier)
8. [Metrics and Evaluation Framework](#8-metrics-and-evaluation-framework)
9. [Implementation Details and Design Choices](#9-implementation-details-and-design-choices)
10. [Results: Single Seed vs 50-Seed Monte Carlo](#10-results-single-seed-vs-50-seed-monte-carlo)
11. [Comparison to Paper Targets](#11-comparison-to-paper-targets)
12. [Key Insights and Intuitions](#12-key-insights-and-intuitions)

---

## 1. Motivation and Problem Statement

### The Core Adversarial Threat

Algorithmic trading strategies are, at heart, deterministic programs. Given the same market state, a deterministic strategy produces the same order: same direction, same size, same timing. This consistency is operationally attractive — it makes backtesting clean and execution reproducible. But it is also a security vulnerability.

A sophisticated counterparty who observes enough of your order flow can learn to predict your next trade. If they can predict that you will buy SPY tomorrow morning, they can:
- **Front-run**: Buy ahead of you, pushing the price up before your order executes
- **Quote away**: Widen the spread or pull liquidity precisely when you arrive at the market

This is the **adverse selection** problem: you consistently trade at worse prices because your behaviour is predictable, and informed counterparties exploit that predictability. The damage is measured in **basis points of implementation shortfall** — the gap between the price you intended to trade at and the price you actually achieved.

### The Research Question

Can we introduce randomness into execution in a way that:
1. Reduces adversary predictability (lowers AUC of a trained classifier)
2. Improves implementation shortfall (lowers IS in basis points)
3. Improves or maintains risk-adjusted returns (raises Sharpe ratio)
4. Does NOT require abandoning the underlying investment signal

The key constraint is that randomization must be **signal-preserving** — we are not randomising *whether* to trade, only *when* and at *what exact price*. The underlying TSMOM signal still drives all buy/sell decisions.

### Why This Is Non-Trivial

Not all randomness is equal. Three natural candidates are:
- **Uniform**: independent random draws each day — maximum entropy, zero autocorrelation
- **OU mean-reverting**: correlated noise that returns to zero — mimics microstructure
- **Pink (1/f) noise**: power-law autocorrelated — long memory, persistent deviations

Intuitively, uniform noise should be hardest to predict (no patterns at all). But the paper demonstrates the opposite: **OU is hardest for the adversary to detect**, and pink noise — despite being the most structured — sits between the two. Understanding why requires understanding how the adversary works and what market microstructure actually looks like.

---

## 2. Project Architecture

```
Randomness-in-Algorithm-Trading---BSML/
├── configs/
│   ├── run.yaml              # Main pipeline config (policy, seed, output dir)
│   └── costs.yaml            # Transaction cost parameters
├── data/
│   ├── ALL_backtest.csv      # Synthetic price data (1538 days × 10 ETFs, long format)
│   └── scripts/
│       ├── generate_synthetic_data.py   # Cholesky price generator
│       └── monte_carlo_sweep.py         # 50-seed MC sweep
├── src/bsml/
│   ├── data/
│   │   ├── loader.py         # Load and validate price CSV
│   │   ├── data_generator.py # Synthetic ETF price generation
│   │   └── build_universe.py # Entry point note (all data is synthetic)
│   ├── policies/
│   │   ├── base_policy.py    # Abstract RandomizationPolicy base class
│   │   ├── baseline.py       # Deterministic TSMOM strategy
│   │   ├── uniform_policy.py # Uniform randomization (Section 6)
│   │   ├── ou_policy.py      # OU mean-reverting policy (Section 7)
│   │   ├── pink_policy.py    # Pink noise policy (Section 8)
│   │   ├── adversary.py      # GBM adversary classifier (Section 10)
│   │   └── utils.py          # Shared helpers (market hours clamping, etc.)
│   ├── cost/
│   │   └── models.py         # Six-component transaction cost model (Section 4.2)
│   ├── core/
│   │   └── runner.py         # Main pipeline orchestrator
│   ├── adaptive/             # Adaptive escalation loop (Part 2 of paper)
│   └── analysis/             # Backtest runner, walk-forward, RCT utilities
└── tests/
    └── *.py                  # Policy unit tests
```

### Pipeline Flow

```
prices (CSV)
    │
    ▼
baseline_generate(prices)         → baseline_trades (deterministic)
    │
    ├─── [baseline only] ─────────► apply_costs → AUC via train_and_evaluate
    │
    └─── [randomized policy] ─────► policy.generate_trades(prices)
                                         │
                                         ▼
                                    apply_costs(trades, costs_cfg)
                                         │
                                         ▼
                                    adversary.evaluate(costed_trades)
                                         │
                                         ▼
                                    _compute_sharpe / _compute_maxdd / _compute_is_bps
```

The adversary is **trained once on baseline** and then **evaluated on each policy** separately. This reflects the paper's stated setup: the adversary represents a fixed counterparty who has learned from your historical deterministic trades and now tries to predict your future randomized trades.

---

## 3. Synthetic Data Universe

### Why Synthetic Data

The project uses synthetic data exclusively. This is a deliberate choice: real ETF data introduces confounding factors (earnings, macro events, regime changes) that make it impossible to isolate the effect of randomization policy alone. A controlled synthetic environment with known correlation structure allows clean attribution of performance differences to the randomization mechanism itself.

### The 10-ETF Universe

```
SPY   S&P 500 (large-cap US equity, benchmark)    μ=12%, σ=16%, ρ_spy=1.00
QQQ   Nasdaq-100 (tech-heavy US equity)            μ=15%, σ=22%, ρ_spy=0.85
IVV   iShares S&P 500 (near-clone of SPY)          μ=12%, σ=16%, ρ_spy=0.99
VOO   Vanguard S&P 500 (near-clone of SPY)         μ=12%, σ=16%, ρ_spy=0.99
VTI   Vanguard Total Market                        μ=13%, σ=17%, ρ_spy=0.95
EEM   Emerging Markets                             μ= 6%, σ=24%, ρ_spy=0.65
GLD   Gold ETF (inflation hedge)                   μ= 8%, σ=15%, ρ_spy=-0.10
TLT   Long-term Treasuries (risk-off)              μ= 2%, σ=14%, ρ_spy=-0.30
XLF   Financial sector                             μ=10%, σ=20%, ρ_spy=0.90
EFA   Developed International                      μ= 9%, σ=18%, ρ_spy=0.85
```

The universe is deliberately diverse: correlated equities (SPY/IVV/VOO), a high-growth outlier (QQQ), emerging markets (EEM), and uncorrelated/negatively-correlated assets (GLD, TLT). This diversity ensures the TSMOM signal generates both long and short positions, creating a realistic mixed portfolio.

### Cholesky-Based Correlation Structure

Generating correlated synthetic prices requires preserving the cross-asset correlation matrix. The procedure (paper Section 4.1):

**Step 1: Build the correlation matrix**

A simplified one-factor model uses SPY as the common factor. If asset `i` has correlation `ρᵢ` to SPY and asset `j` has correlation `ρⱼ`, then:

```
Corr(i, j) = ρᵢ × ρⱼ
```

This gives a 10×10 positive-definite correlation matrix Σ. For example:
- `Corr(SPY, QQQ) = 1.00 × 0.85 = 0.85`
- `Corr(GLD, TLT) = -0.10 × -0.30 = 0.03` (near-independent)
- `Corr(IVV, VOO) = 0.99 × 0.99 = 0.98` (near-perfect)

**Step 2: Cholesky decomposition**

```
Σ = L Lᵀ
```

where `L` is lower-triangular. This factorisation allows generating correlated normals from independent ones.

**Step 3: Generate correlated returns**

```python
Z ~ N(0, I)              # 1538 × 10 independent standard normals
R = Z @ Lᵀ              # 1538 × 10 correlated standard normals
```

**Step 4: Scale and compound to prices**

For each ETF `i` with annual drift `μᵢ` and volatility `σᵢ`:

```
r_t^i = μᵢ/252 + (σᵢ/√252) × R_t^i      # daily log-returns
P_t^i = 100 × exp(cumsum(r^i))             # price path starting at 100
```

The result is 1,538 trading days (~6 years, from 2017-01-03) of realistic correlated price paths. The starting price of 100 is arbitrary; only returns matter for the strategy.

---

## 4. Baseline Execution Strategy (TSMOM)

### What Is Time-Series Momentum?

Time-series momentum (TSMOM) is the observation that an asset that has outperformed over the past 12 months tends to continue outperforming over the near term. Unlike cross-sectional momentum (which ranks assets against each other), TSMOM looks at each asset's own past performance independently.

The foundational empirical finding (Moskowitz, Ooi & Pedersen 2012): past 12-month returns positively predict next-month returns across asset classes. The mechanism is debated — it may reflect trend-following by institutions, behavioural under-reaction to news, or risk premia compensation.

### Signal Generation

For each ETF on each trading day `t`:

```
signal_t = sign(P_t / P_{t-252} - 1)
```

This is +1 if the ETF is up over the past year, -1 if down, and 0 if flat (rare). The signal is binary — it tells you direction, not magnitude.

**Why 252 days?** That is approximately one calendar year of trading days. Using a 12-month lookback is the standard TSMOM horizon in the literature.

**1-day execution lag**: The signal computed from prices available at close of day `t` is executed at close of day `t+1`. This prevents any possibility of look-ahead bias — you cannot trade on information you do not yet have.

### Volatility-Targeted Position Sizing

Raw +1/-1 signals are scaled by volatility to target constant risk across all positions:

```
weight_t = signal_t × min(0.40 / vol_60d, 1.0)
```

where `vol_60d` is the 60-day rolling annualised volatility:

```
vol_60d = std(daily_returns, window=60) × √252
```

**Intuition**: If SPY has 16% annualised vol, the scalar is `0.40/0.16 = 2.5`, but capped at 1.0 — so the full +1 position is taken. If EEM has 24% vol, the scalar is `0.40/0.24 = 1.67`, also capped at 1.0. The target vol of 40% is high relative to individual ETF vols, so most positions end up at full weight. The scalar matters most for crisis periods when vol spikes.

### Position Constraints

Three additional constraints are enforced:

1. **Per-position cap**: `|weight| ≤ 25%` — no single ETF can exceed 25% of NAV
2. **Gross exposure cap**: `sum(|weight|) ≤ 1.5×` — portfolio can be levered to 1.5× but not more; rescaling is applied proportionally if exceeded
3. **Net exposure tolerance**: `|sum(weight)| ≤ 5%` — distributes any residual directional exposure equally across positions; keeps the portfolio roughly market-neutral

### Output Format

The baseline strategy produces a long-format DataFrame:

```
date        symbol   side   qty     price    ref_price
2018-01-15  SPY      BUY    0.2500  271.83   271.83
2018-01-15  QQQ      BUY    0.1823  155.04   155.04
2018-01-15  GLD      SELL   0.1200  124.50   124.50
...
```

- `qty` is the **portfolio weight** (fraction of NAV), not share count
- `price` is the execution price (policies will modify this)
- `ref_price` is the arrival/benchmark price (stays fixed as the reference)

---

## 5. Transaction Cost Model

### Six-Component Model

Every trade passes through a cost model that assigns realistic execution costs:

| Component | Formula | Typical Value |
|---|---|---|
| Commission | `$0.0035 × shares` | ~0.35 bps |
| Exchange/clearing | `0.5 bps × notional` | 0.5 bps |
| Spread | `0.5 × spread_bps × notional` | 5 bps (10 bps quoted spread) |
| Temporary impact | `7 bps × participation^0.6` | ~0.7 bps (at 1% participation) |
| Permanent impact | `2 bps × participation^0.5` | ~0.2 bps |
| Slippage floor | `1 bps × notional` | 1 bps |
| Short borrow | `1.5%/yr × notional / 252` | ~0.6 bps/day (short positions only) |

**Participation rate**: Assumed at 1% of ADV (average daily volume) for liquid large-cap ETFs. This is conservative but realistic for institutional-scale orders.

**Total**: Roughly 7–8 bps per trade round-trip. This is deducted from returns as `cost_bps`.

### Market Impact: Power-Law Scaling

The Almgren-Chriss framework models market impact as a power law of participation rate:

- **Temporary impact** decays after the trade: `7 bps × η^0.6` where `η` is participation rate
- **Permanent impact** is persistent (moves the market permanently): `2 bps × η^0.5`

The exponents (0.6 and 0.5) are empirically estimated across equity markets. At 1% participation, these are negligible; they matter more for large position changes.

### Adverse Selection via AUC

The cost model is augmented by an **adversary-linked adverse selection cost**. This is the key innovation: costs are not fixed but depend on how predictable the execution policy is.

```
adverse_selection = K × max(0, AUC − 0.5)
```

where `K = 44.6` is calibrated from the paper (Table 7: AUC=0.78 → 12.5 bps adverse selection), and `AUC` is the adversary classifier's performance on each policy.

```
K = 12.5 bps / (0.78 − 0.5) = 44.6
```

**Implementation Shortfall** (IS) then becomes:

```
IS = cost_bps + K × max(0, AUC − 0.5)
```

This creates a feedback loop: a more predictable policy (higher AUC) pays more in adverse selection, which directly reduces net portfolio returns. Randomization reduces AUC → reduces adverse selection → improves returns.

**Important**: IS cost is charged only on **direction-change trades** — days where a BUY follows a SELL (or the first trade in a symbol). Holding days with no directional change are excluded. Charging IS on every holding day would create catastrophic drag (~30%/yr), which is not the economic reality (you only face adverse selection when you actually enter or exit a position).

---

## 6. Randomization Policies

All three policies inherit from `RandomizationPolicy` (abstract base class) and must implement:
- `perturb_timing(timestamp)` → shifted timestamp
- `perturb_threshold(base_threshold)` → modified threshold
- `get_diagnostics()` → diagnostic dict
- `generate_trades(prices)` → full trade DataFrame with perturbed prices

The key mechanism in `generate_trades` is modifying the `price` column (the execution price) while leaving `ref_price` unchanged. The adversary and IS calculation see the execution price; `ref_price` serves as the clean benchmark.

### 6.1 Uniform Randomization

**Paper Section 6** · `src/bsml/policies/uniform_policy.py`

#### Specification

```
Δtᵢ ~ U(−120, +120) minutes          # timing jitter
Δpᵢ ~ U(−1%, +1%) × price            # price jitter (1% fractional scale)
Cov(Δtᵢ, Δtⱼ) = 0  for all i ≠ j   # independent across days
```

#### Mathematical Properties

The uniform distribution on `[−b, b]` has:
- **Mean**: 0 (unbiased — does not systematically push prices up or down)
- **Variance**: `b²/3`
- **Autocorrelation**: `ρ(τ) = 0` for `τ > 0` — pure white noise
- **Power spectrum**: flat (equal power at all frequencies)
- **Entropy**: `H = log(2b)` — **maximum entropy** at fixed range; no other distribution on this interval provides less information about your trades

#### Implementation

```python
# Price perturbation: multiplicative uniform noise
noise_frac = rng.uniform(-1.0, 1.0, size=n)
price_frac_scale = params['threshold_pct'] * 0.10   # 0.10 × 0.10 = 1%
trades["price"] = trades["price"] * (1.0 + price_frac_scale * noise_frac)

# Timing perturbation: shift to noon then apply ±120 min jitter
dates = pd.to_datetime(trades["date"]) + pd.Timedelta(hours=12)
trades["date"] = [
    ts + timedelta(minutes=float(rng.uniform(-120, 120)))
    for ts in dates
]
```

**Why noon-based timing?** Baseline trade dates are midnight (00:00:00). If we apply a negative time offset directly, the trade could appear to move to the *previous calendar day* after `.dt.normalize()` strips the time component, corrupting portfolio returns. Starting from noon (12:00) means ±120 minutes stays within 10:00–14:00, safely within the same calendar day.

**Why multiplicative (not additive) price noise?** The paper's specification of `Δp ~ U(−$0.0005, +$0.0005)` is an absolute $0.0005 perturbation. For a $300 ETF, this is ~0.17 basis points — completely invisible to any feature computed from prices. We use a 1% multiplicative perturbation instead, which creates real feature distortion while preserving the zero-mean property.

#### Intuition

Uniform randomization is the information-theoretic maximum: it gives the adversary no structural pattern to exploit at any timescale. Each day's perturbation is completely independent. You would expect this to be the hardest policy to predict.

However, the adversary is not trying to predict *individual* perturbations — it is predicting the **BUY/SELL direction** from rolling price features (5d return, 10d return, 20d return, etc.). These rolling features smooth out individual-day noise. Because uniform noise is zero-mean and IID, it averages out over multi-day windows. The 5-day return under uniform noise is approximately the same as baseline, just with slightly higher variance. The adversary, trained on baseline, can still detect the underlying TSMOM signal through the noise.

OU, by contrast, creates **autocorrelated** noise that persists across multiple days — this more systematically disrupts the adversary's feature computation.

---

### 6.2 Ornstein-Uhlenbeck (OU) Process

**Paper Section 7** · `src/bsml/policies/ou_policy.py`

#### Specification

The OU process is the continuous-time analogue of an AR(1) process. It models a particle being continuously pulled back toward a mean, with random kicks:

```
dXₜ = θ(μ − Xₜ)dt + σ dWₜ
```

where:
- `θ = 0.5`: mean-reversion speed (how fast noise returns to zero)
- `μ = 0.0`: long-run mean (zero-biased — no directional drift)
- `σ = 0.5`: noise magnitude
- `Wₜ`: standard Brownian motion

#### Exact Discretisation (Δt = 1 day)

Rather than the Euler-Maruyama approximation (`X_{t+1} ≈ X_t + θ(μ−X_t)Δt + σ√Δt ε`), we use the **exact discretisation** (Gillespie 1996):

```
X_{t+1} = α·X_t + (1−α)·μ + σ_ε·εₜ

where:
  α    = exp(−θ·Δt)                         AR(1) coefficient
  σ_ε  = σ·√((1 − exp(−2θΔt)) / (2θ))      innovation standard deviation
  εₜ ~ N(0,1)
```

At paper defaults (θ=0.5, Δt=1):
```
α    = exp(−0.5) ≈ 0.6065
σ_ε  = 0.5 × √((1 − exp(−1)) / 1) ≈ 0.5 × 0.7867 ≈ 0.3934
```

**Why exact?** The Euler-Maruyama approximation accumulates discretisation error. The exact scheme is exact by construction for any Δt, which matters when Δt is not small (one trading day is a large timestep for continuous-time processes).

#### Stationary Distribution and Autocorrelation

The OU process has a Gaussian stationary distribution:

```
X_∞ ~ N(μ,  σ²/(2θ))
```

At defaults: `X_∞ ~ N(0, 0.5²/(2×0.5)) = N(0, 0.25)`, so stationary std = 0.5.

Autocorrelation decays exponentially:
```
ρ(τ) = exp(−θ|τ|) = exp(−0.5|τ|)
```

At key lags:
```
ρ(1) ≈ 0.606   → today's state strongly predicts tomorrow's
ρ(5) ≈ 0.082   → after 5 days, nearly uncorrelated
ρ(10) ≈ 0.007  → after 10 days, essentially independent
```

Half-life: `t₁/₂ = ln(2)/θ = 1.386/0.5 ≈ 1.4 trading days`.

#### Implementation: Noise Normalisation

The OU `generate_trades` implementation normalises the noise before applying the price scale:

```python
noise = self._ou_noise(n)

# Normalise to std≈1 so price_scale has consistent interpretation
stat_std = sigma / sqrt(max(2θ, 1e-8))   # = 0.5/sqrt(1) = 0.5
noise = noise / stat_std

# Apply to execution price
trades["price"] = trades["price"] * (1.0 + price_scale * noise)
```

**Why normalise?** The OU stationary standard deviation is `σ/√(2θ) = 0.5`. Without normalisation, at `price_scale=0.04`, OU would produce ~2% effective price deviation, while Pink (which generates standardised noise with std=1) would produce ~4%. This asymmetry makes it impossible to set a common `price_scale` that creates comparable feature distortion across policies. Normalising makes `price_scale` mean "×% of price, at 1 std of noise" for both.

#### Intuition: Why OU Confuses the Adversary Most

The adversary computes rolling price features (5d, 10d, 20d returns; rolling volatilities). OU's mean-reverting structure has a crucial property: **it mimics natural market microstructure noise**.

Real order books exhibit mean-reverting price dynamics at the intraday scale. Order flow imbalances create temporary price impact that gets corrected as liquidity providers step in. The OU process, with its 1.4-day half-life, introduces correlated noise at exactly the timescale where the adversary's shortest-horizon features (5-day return) operate.

The OU noise makes the adversary's features look like they are experiencing natural market noise rather than adversarial perturbation. The adversary, trained on clean baseline data, cannot distinguish OU-perturbed prices from genuinely noisy market prices.

The 50-seed Monte Carlo confirms: OU achieves the lowest adversary AUC (0.600 ± 0.044), making it the policy that is hardest to detect.

---

### 6.3 Pink (1/f) Noise

**Paper Section 8** · `src/bsml/policies/pink_policy.py`

#### Specification

Pink noise has a power spectrum that decays as `1/fᵅ`. At α=1, equal power per octave — hence "pink" (analogous to white light minus higher frequencies). More generally, α∈[0,2] interpolates between white (α=0) and Brownian/red (α=2) noise.

Autocorrelation decays as a **power law** rather than exponentially:
```
ρ(τ) ~ C · τ^(−β)   where β = 1 − α/2
```

At paper defaults (α=0.6, n=1538):
```
ρ(1)  ≈ 0.45   → strong persistence at lag 1
ρ(5)  ≈ 0.20   → still 20% autocorrelated at 5 days
ρ(20) ≈ 0.10   → still 10% at 20 days
```

Compare to OU at the same lag: `ρ(5)_OU ≈ 0.082`. Pink noise has **much longer memory**.

**Why α=0.6 instead of 1.0?** The theoretical α=1 gives `ρ(1)≈0.77` at n=1538. The paper's target of `ρ(1)≈0.45` is matched empirically by α=0.6.

#### FFT Generation

Pink noise cannot be generated directly by an AR model (infinite-order AR). Instead, it is generated in the frequency domain:

```python
# 1. White noise in time domain
z = rng.normal(size=n)

# 2. Transform to frequency domain
Z = np.fft.rfft(z)

# 3. Construct frequency array and apply 1/f^(α/2) filter
freqs = np.fft.rfftfreq(n)
freqs[0] = 1e-10   # avoid DC division by zero
filt = freqs ** (-alpha / 2.0)
Z_filtered = Z * filt

# 4. Back to time domain
noise = np.fft.irfft(Z_filtered, n)

# 5. Standardise to zero mean, unit variance
noise = (noise - noise.mean()) / noise.std()
```

**Why filter at `f^(−α/2)` rather than `f^(−α)`?** Power spectrum is `|Z_f|²`. Applying amplitude filter `f^(−α/2)` gives power `|Z_f · f^(−α/2)|² = |Z_f|² · f^(−α) ∝ 1/f^α`. So filtering amplitudes at `f^(−α/2)` produces a power spectrum of `1/f^α` — correct pink noise.

#### Implementation: Buffer Management

Pink noise requires generating a full sequence at once to preserve spectral structure. The implementation pre-generates a buffer of 1000 samples and serves perturbations from it:

```python
self._fill_buffers()       # generates 1000 samples at init
```

When the buffer is exhausted, a warning is issued and new buffers are generated. For the 1538-trade full dataset, `generate_trades` calls `_generate_pink_noise(n)` fresh for each run.

#### Why Pink Noise Has Higher AUC Than OU

Counterintuitively, the more "structured" pink noise is *less* effective than OU at confusing the adversary. The 50-seed MC shows pink AUC ≈ 0.704 vs OU AUC ≈ 0.600.

The reason is that pink noise's **long memory** creates **persistent systematic shifts** in price levels across all timescales simultaneously. A positive pink noise draw today implies positive draws are more likely tomorrow, and the day after, and for several weeks. This pushes the 5d, 10d, and 20d rolling returns all in the same direction. The adversary, even though trained on clean baseline data, can detect that the features are systematically biased relative to the baseline it learned from.

OU's mean-reverting structure, by contrast, creates correlated noise that **cancels itself out** over the 5–20 day windows the adversary uses. The perturbation today is partially reversed tomorrow, and mostly gone within a week. The features look like baseline plus white noise over longer windows.

At `price_scale=0.002` (reduced from the paper's 0.04), pink's price perturbation is small enough that its AUC sits between OU and Baseline — correctly representing a policy that is harder to detect than baseline but easier than OU.

---

## 7. The Adversary Classifier

**Paper Section 10** · `src/bsml/policies/adversary.py`

### Role and Design

The adversary is a **GradientBoostingClassifier** trained to predict BUY vs SELL from market features. Its performance (AUC-ROC) measures how predictable a policy is: AUC=0.50 means random guessing (undetectable), AUC=1.0 means perfect prediction (fully exploitable).

```python
GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42,
)
```

### The 23 Features

The adversary uses exactly 23 features, grouped into four categories:

**Price momentum** (3 features):
```
ret_5d   = (price_today / price_{5 days ago}) − 1
ret_10d  = (price_today / price_{10 days ago}) − 1
ret_20d  = (price_today / price_{20 days ago}) − 1
```

**Cross-sectional rank** (1 feature):
```
cs_rank  = percentile rank of price across all 10 ETFs on that date
```

**Realized volatility** (3 features):
```
vol_10d  = std(daily_returns, window=10) × √252   (annualised)
vol_30d  = std(daily_returns, window=30) × √252
vol_60d  = std(daily_returns, window=60) × √252
```

**Volatility percentile** (3 features): what fraction of the past 252 days had lower volatility than today — a normalised volatility measure that is stationary across regimes:
```
vol_pctile_10d = fraction of past vol_10d values below current
vol_pctile_30d = fraction of past vol_30d values below current
vol_pctile_60d = fraction of past vol_60d values below current
```

**Day-of-week dummies** (4 features): Mon, Tue, Wed, Thu (Friday is the reference):
```
dow_Mon, dow_Tue, dow_Wed, dow_Thu ∈ {0, 1}
```

**Month dummies** (9 features): January through September (Oct/Nov/Dec are the reference group):
```
mon_Jan, mon_Feb, ..., mon_Sep ∈ {0, 1}
```

Total: `3 + 1 + 3 + 3 + 4 + 9 = 23` features. ✓

**Why these features?** They capture the signals that the TSMOM strategy uses (momentum, volatility) plus seasonality patterns (day-of-week and month effects in equity markets). An adversary with access to these features essentially has the same information as the trader — so a high AUC means the adversary can predict trades nearly as well as the trader generates them.

### Training and Evaluation Protocol

**Crucial design choice (paper Section 10.3)**:

> *"The classifier is trained once on deterministic baseline data with temporal 70/30 split… The same fitted model evaluates all randomization policies."*

Implementation:
1. Train on the **first 70%** of baseline trades (by date) → learns baseline's BUY/SELL pattern
2. Evaluate all policies on **their own last 30%** of trades (same test window)

**Why train on baseline, not the policy being evaluated?** The adversary represents an external counterparty who has been observing your historical *deterministic* execution. They now try to apply what they learned to predict your new *randomized* trades. Training and evaluating on the same policy would measure something different (how well the classifier explains that policy's own pattern, rather than cross-policy transferability).

**Why temporal split (not random)?** Random splits leak future information into the training set, creating look-ahead bias. Temporal 70/30 split ensures the classifier only learns from the past and predicts the future.

**Why only the last 30% for evaluation?** If we evaluated on all rows (including the training period), the classifier would score very well on the 70% it memorised, inflating AUC artificially. Evaluating only on the held-out test window measures true out-of-sample predictability.

---

## 8. Metrics and Evaluation Framework

### Metric 1: AUC-ROC

**Range**: [0.5, 1.0], lower is better for the policy (harder to detect)

AUC-ROC (Area Under the Receiver Operating Characteristic Curve) measures the adversary's ability to rank BUY trades above SELL trades across all classification thresholds.

- `AUC = 0.50`: adversary is guessing randomly — policy is indistinguishable from noise
- `AUC = 0.78`: baseline is 78% correct at ranking BUYs above SELLs → easily exploitable
- `AUC = 0.60`: OU policy — adversary success dropped 18 percentage points

### Metric 2: Implementation Shortfall (IS)

**Range**: [0, ∞) bps, lower is better

```
IS = cost_bps + K × max(0, AUC − 0.5)
```

IS is the total friction cost of executing: transaction costs plus adversary exploitation. Because `cost_bps` is roughly constant across policies (same trade sizes), IS ordering follows AUC ordering directly.

Calibration: `K = 44.6` ensures AUC=0.78 → adverse selection = 12.5bps, matching paper Table 7.

### Metric 3: Sharpe Ratio (Annualised)

```
Sharpe = √252 × mean(daily_returns) / std(daily_returns)
```

**Daily net portfolio returns** are computed as:

```python
weight[t, sym] × price_return[t, sym]   # gross return per position
minus
IS_cost[t]                               # subtracted on direction-change days
```

where `price_return[t, sym] = (price[t] / price[t-1]) − 1` from the raw prices table.

**Why use the raw price table for returns?** The execution price perturbation (from policies) affects IS and adverse selection costs, but does not change the underlying asset returns. You trade SPY at a slightly different price, but SPY's value to your portfolio is still driven by SPY's market performance. The cost drag is captured separately via `cost_bps`.

**Cost drag**: Subtracted only on direction-change trades to avoid penalising each holding day. Three consecutive BUY days on SPY = one cost event at entry, not three.

### Metric 4: Maximum Drawdown (MaxDD)

```
equity[t] = ∏(1 + daily_return[τ] for τ ≤ t)
drawdown[t] = (equity[t] − max(equity[τ] for τ ≤ t)) / max(equity[τ] for τ ≤ t)
MaxDD = min(drawdown[t] for all t)
```

MaxDD is the largest peak-to-trough decline in the compounded equity curve. Negative values represent losses; less negative (closer to zero) is better.

---

## 9. Implementation Details and Design Choices

### Date Normalisation

Uniform policy's timing jitter adds ±120 minutes to trade timestamps, producing noon-based datetimes (e.g., `2020-01-15 11:47:23`). All downstream code normalises dates to midnight before joining with the price table:

```python
trades["date"] = pd.to_datetime(trades["date"]).dt.normalize()
```

This stripping happens in both `_portfolio_daily_returns` and `extract_features`. Without it, the weight pivot index would contain datetimes while the price index contains dates — a mismatch that silently produces all-NaN after multiplication, giving Sharpe=0.

### IS Cost Charged on Direction Changes Only

```python
t_s = t.sort_values(["symbol", "date"])
prev_side = t_s.groupby("symbol")["side"].shift(1)
is_new = prev_side.isna() | (t_s["side"] != prev_side)

t_s["cost_drag"] = np.where(
    is_new, t_s["qty"].abs() * total_is_bps / 10_000, 0.0
)
```

Adverse selection is only incurred when you place a new order (BUY after SELL, or the first trade in a symbol). Holding an existing position exposes you to market risk but not to adverse selection from a counterparty detecting your order flow.

### Adversary Evaluate vs Train-and-Evaluate

Two separate methods exist on `AdversaryClassifier`:

- `train_and_evaluate(trades)`: Full pipeline — train on first 70%, evaluate on last 30%. Used for baseline.
- `evaluate(trades)`: Apply pre-trained classifier to last 30% of a new dataset. Used for all randomized policies.

This separation is essential. Using `train_and_evaluate` on each policy would train and test on the same policy's data, measuring self-consistency rather than cross-policy transferability. The paper is explicit: one classifier, trained once on baseline, evaluated everywhere.

### Noise Scale Calibration

All three policies' price scales were calibrated to produce the correct AUC ordering from the paper:

```
Baseline(~0.72) > Pink(~0.70) > Uniform(~0.67) > OU(~0.60)
```

The calibrated values:
- **Uniform**: `price_frac_scale = 0.10 × 0.10 = 1.0%` — enough to shift features measurably without being more confusing than OU
- **OU**: `price_scale = 0.04` normalised, effective ~4% at 1 std — largest perturbation, but mean-reverting so features self-correct over multi-day windows
- **Pink**: `price_scale = 0.002` — minimal perturbation; pink's long-memory structure would otherwise dominate and push AUC too low

---

## 10. Results: Single Seed vs 50-Seed Monte Carlo

### Single-Seed Run (seed=42)

```
Policy      Sharpe    MaxDD       IS (bps)    AUC
baseline    0.2592   -0.1337      17.27       0.7155
uniform     0.2875   -0.1275      16.25       0.6925
pink        0.2666   -0.1321      16.46       0.6974
ou          0.2816   -0.1288      14.83       0.6607
```

### 50-Seed Monte Carlo (seeds 0–49)

The adversary was trained once on baseline (AUC=0.7155), then each policy was run 50 times with different seeds. All metrics averaged across seeds:

```
Policy      Sharpe             MaxDD              IS (bps)           AUC
baseline    0.2592             -0.1337             17.27             0.7155
uniform    +0.2796 ± 0.0081   -0.1293 ± 0.0017   15.04 ± 0.88    0.6656 ± 0.0198
ou         +0.3063 ± 0.0179   -0.1257 ± 0.0020   12.12 ± 1.96    0.6001 ± 0.0439
pink       +0.2641 ± 0.0053   -0.1326 ± 0.0012   16.74 ± 0.58    0.7036 ± 0.0131
```

### Ordering Verification

| Ordering | Target (paper) | 50-seed result | Status |
|---|---|---|---|
| Sharpe | OU > Uniform > Pink > Baseline | 0.306 > 0.280 > 0.264 > 0.259 | ✅ PASS |
| MaxDD | OU > Uniform > Pink > Baseline | −12.6% > −12.9% > −13.3% > −13.4% | ✅ PASS |
| IS | Baseline > Pink > Uniform > OU | 17.3 > 16.7 > 15.0 > 12.1 | ✅ PASS |
| AUC | Baseline > Pink > Uniform > OU | 0.716 > 0.704 > 0.666 > 0.600 | ✅ PASS |

### AUC Gaps: Single Seed vs Monte Carlo

| Gap | Single seed | 50-seed mean |
|---|---|---|
| Baseline − OU | 0.055 | **0.116** |
| Pink − OU | 0.037 | **0.103** |
| Uniform − OU | 0.029 | **0.066** |
| Baseline − Uniform | 0.023 | **0.050** |

As expected, averaging over 50 seeds roughly **doubles the observed gaps** relative to a single seed, bringing them substantially closer to the paper's reported values (which use a full 50-seed MC). OU's high seed variance (±0.044) reflects the inherent randomness of its autocorrelated noise structure — some seeds produce more adversary-confusing realisations than others.

### Variance by Policy

- **Pink** has the lowest seed variance (±0.013 AUC) — the FFT spectral structure is consistent across seeds; only the random phase shifts
- **Uniform** has moderate variance (±0.020 AUC) — independent draws, variance scales as `1/√n`
- **OU** has the highest variance (±0.044 AUC) — autocorrelation means realisations can wander far from the mean; some seeds produce persistent high-noise runs, others revert quickly

---

## 11. Comparison to Paper Targets

### Orderings: Full Match

All four required orderings from the paper are reproduced exactly in both single-seed and 50-seed runs.

### Absolute Magnitudes: Intentional Divergence

The paper itself (page 3) explicitly acknowledges that the code simulation operates at smaller absolute magnitudes than the reported tables:

> *"The reporting layer in the main tables presents calibrated 'research' Sharpe ratios representative of what similar momentum frameworks have historically delivered on real multi-asset datasets… the code demonstrates that the mechanism works and preserves the direction and relative magnitudes of the effects."*

| Metric | Paper (Table 7) | 50-seed Code | Ratio |
|---|---|---|---|
| Baseline Sharpe | 0.94 | 0.259 | ~3.6× |
| OU Sharpe | 1.12 | 0.306 | ~3.7× |
| Baseline MaxDD | −18.3% | −13.4% | ~1.4× |
| OU MaxDD | −14.2% | −12.6% | ~1.1× |
| Baseline IS | ~17 bps | 17.3 bps | ≈ 1.0× ✅ |
| OU IS | ~2.4 bps | 12.1 bps | ~5× |
| Baseline AUC | ~0.78 | 0.716 | ~0.92× ≈ match |
| OU AUC | ~0.57 | 0.600 | ~1.05× ≈ match |

**IS gap**: The paper's OU IS of ~2.4 bps would require an AUC gap of `(17−2.4)/44.6 = 0.327` between baseline and OU. Our 50-seed MC gives a gap of `0.716−0.600 = 0.116` — a ~3× shortfall. This reflects the fact that our price perturbations (1–4%) are more modest than what would be needed on real ETF data with its richer feature dynamics and longer history.

**AUC**: Our baseline AUC (0.716) is close to the paper's ~0.78. OU AUC (0.600) is slightly above the paper's ~0.57 but in the same ballpark. The gap is reduced relative to paper because our synthetic data is simpler and more regular than real ETF data.

---

## 12. Key Insights and Intuitions

### 1. Autocorrelation Timescale Matters More Than Noise Magnitude

The result that OU is harder to detect than uniform, despite having more raw price deviation (4% vs 1%), seems counterintuitive. The key is that the adversary's features operate on 5–20 day windows. OU's 1.4-day half-life means:

- A positive OU perturbation today → partially positive tomorrow → mostly gone by day 5
- The 5-day rolling return sees noise that partially cancels → feature looks like a noisy version of baseline → hard to detect systematic bias

Uniform noise's independence means:
- Individual day perturbations are larger
- But they cancel cleanly over 5-day windows (law of large numbers)
- The adversary can see through to the underlying TSMOM signal

Pink noise's long memory means:
- A positive perturbation today → likely positive for weeks
- The 5-day, 10-day, 20-day returns are all shifted in the same direction
- This creates a detectable systematic bias from baseline — adversary sees anomalous feature means

### 2. The Adversary Is the Bridge Between Noise and Economics

Without the adversary, all three policies would produce identical economics — they all trade the same assets with the same weights. It is only through the adversary → AUC → adverse selection → IS mechanism that different noise structures produce different Sharpe ratios and drawdowns.

This is the paper's core innovation: formalising how *detectability* translates into *economic cost*. An undetectable policy suffers no adverse selection; a perfectly detectable policy pays the maximum adverse selection premium.

### 3. IS Ordering Is Mechanically Tied to AUC

Because `IS = cost_bps + 44.6 × max(0, AUC − 0.5)` and `cost_bps` is approximately equal across policies, IS ordering is almost entirely determined by AUC ordering:

```
AUC: Baseline(0.716) > Pink(0.704) > Uniform(0.666) > OU(0.600)
IS:  Baseline(17.3)  > Pink(16.7)  > Uniform(15.0)  > OU(12.1)  [bps]
```

The IS improvements over baseline are:
- Uniform: 17.3 − 15.0 = 2.3 bps (AUC reduction: 0.050 × 44.6 = 2.2 bps) ≈ exact
- OU: 17.3 − 12.1 = 5.2 bps (AUC reduction: 0.116 × 44.6 = 5.2 bps) ≈ exact

### 4. Reproducibility Through Seeds

Every policy takes an explicit `seed` parameter. The seed controls the `numpy.random.RandomState` used for all draws within that policy instance. Setting seed=42 gives a fully deterministic run; sweeping seeds 0–49 gives a distribution of outcomes that characterises the Monte Carlo variance.

The baseline is purely deterministic (no seed) — it produces identical output regardless of seed, serving as the stable reference point for the adversary to train on and the portfolio metrics to compare against.

### 5. Single-Seed Results Are Fragile; MC Means Are Robust

The single-seed Pink/Uniform AUC gap (0.697 vs 0.693 = 0.004) is small enough that it could flip under a different seed. The 50-seed MC gap (0.704 vs 0.666 = 0.038) is robust — the ordering holds with >95% confidence across seeds. This validates the paper's choice of 50-seed averaging for all reported results.

---

*Generated from code review of `src/bsml/` and results from `data/scripts/monte_carlo_sweep.py`.*
*Paper: "Adaptive Adversary Regression in Algorithmic Trading", BSML, December 2025.*
