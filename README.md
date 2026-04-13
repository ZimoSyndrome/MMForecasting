# Multi-Model Market Forecasting & Risk Engine

A quantitative research project that forecasts equity log-returns by combining autoregressive price dynamics with a broad macro feature matrix sourced from multiple government data agencies. Three forecasting paradigms are evaluated under strict walk-forward conditions, with emphasis on statistical validity, no look-ahead bias, and honest out-of-sample measurement.

---

## Models

| Model | Class | Mean equation | Variance |
|---|---|---|---|
| **ARIMAX-GARCH** | Econometric | Lagged returns + macro lags (OLS) | GARCH(1,1) on residuals |
| **PCA+EN** | Statistical ML | PCA-reduced macro matrix → ElasticNet | — |
| **PLS** | Latent-factor | Partial Least Squares (latent components) | — |
| **XGBoost** | ML (tree) | Gradient-boosted trees over full feature matrix | — |
| **LSTM** | Deep Learning | Sequence model over price + macro context | — |

---

## Mathematical Formulation

### 1. Return series

$$r_t = \log\!\left(\frac{P_t}{P_{t-1}}\right)$$

All models target the daily log-return $r_t$. The series is tested for stationarity (ADF) before model fitting.

### 2. Baseline ARIMA(p\*,d\*,q\*)-GARCH(1,1)

The baseline mean equation is selected by AIC grid search over $(p, d, q) \in \{0..4\} \times \{0..2\} \times \{0..4\}$:

$$r_t^{(d)} = \mu + \sum_{i=1}^{p^*} \phi_i r_{t-i}^{(d)} + \sum_{j=1}^{q^*} \theta_j \varepsilon_{t-j} + \varepsilon_t$$

where $r_t^{(d)}$ denotes the $d^*$-times differenced return. **No exogenous regressors** enter this baseline — it is a pure time-series model. The GARCH distribution (Normal / Student-t / Skewed-t / GED) is also AIC-selected.

**Special case — ARIMA(0,0,0):** When AIC selects $(p^*,d^*,q^*) = (0,0,0)$, the mean equation collapses to a constant:

$$r_t = \mu + \varepsilon_t \quad \Longrightarrow \quad \text{intercept-only-GARCH(1,1)}$$

This is the degenerate "random walk with drift" mean assumption — the model attributes all return variation to the conditional variance process, not to predictable linear structure.

### 3. ARX-GARCH two-step (PCA+EN and PLS)

The macro-augmented models follow a two-step procedure:

**Step 1 — Mean equation with exogenous macro lags:**

$$r_t = \mu + \mathbf{x}_{t-1}^\top \boldsymbol{\beta} + \varepsilon_t$$

where $\mathbf{x}_{t-1}$ contains $K \times L_{\max}$ lagged macro features (strictly $t-1$ or earlier — no contemporaneous regressors). The coefficient vector $\boldsymbol{\beta}$ is estimated via PCA+ElasticNet or PLS.

**Degenerate case:** If ElasticNet regularisation shrinks $\boldsymbol{\beta} \to \mathbf{0}$ (all macro and AR coefficients zeroed), the model is again labelled **intercept-only-GARCH(1,1)**. Economically this means the macro features, as of that training window, carry no statistically reliable signal.

**Step 2 — Variance equation on residuals:**

$$\hat{\varepsilon}_t = r_t - \hat{r}_t$$

$$\sigma_t^2 = \omega + \alpha \hat{\varepsilon}_{t-1}^2 + \beta \sigma_{t-1}^2, \quad \alpha + \beta < 1$$

Fitted with `mean="Zero"` (the residual series is already mean-zero by construction). This ensures $\sigma_t^2$ captures only the heteroskedastic component of volatility, not any residual level shift.

### 4. GARCH(1,1) — explicit parameter constraints

| Parameter | Interpretation | Constraint |
|---|---|---|
| $\omega > 0$ | Long-run variance floor | $\omega > 0$ |
| $\alpha \geq 0$ | ARCH term — shock sensitivity | $\alpha \geq 0$ |
| $\beta \geq 0$ | GARCH term — variance persistence | $\beta \geq 0$ |
| — | Stationarity (finite unconditional variance) | $\alpha + \beta < 1$ |
| — | Unconditional variance | $\bar{\sigma}^2 = \omega / (1 - \alpha - \beta)$ |

### 5. Macro feature matrix

Let $\mathcal{M}$ be the set of $K$ macro series, each observed at native frequency $f_k$ with publication lag $\ell_k$ calendar days. For each series $k$:

1. **Stationarity transform** at native frequency (see table below)
2. **Release-lag shift**: index shifted forward by $\ell_k$ to prevent look-ahead:
   $$\tilde{t}_k = t_k^{\text{ref}} + \ell_k$$
3. **Forward-fill** to daily trading calendar (limit = $\delta_k$ trading days by frequency)
4. **Lag construction**: for each lag $L = 1, \ldots, L_{\max}$, feature $x_{k,t}^{(L)} = \tilde{z}_{k,t-L}$

The exogenous design matrix is:

$$X_{\text{exog}} \in \mathbb{R}^{T \times (K \cdot L_{\max})}$$

### 6. PCA+ElasticNet

Principal components $\mathbf{V} \in \mathbb{R}^{(K L_{\max}) \times d}$ extracted from $X_{\text{exog}}$:

$$\mathbf{Z} = X_{\text{exog}} \mathbf{V}, \quad \mathbf{Z} \in \mathbb{R}^{T \times d}$$

ElasticNet fitted on $[\mathbf{Z}, X_{\text{AR}}]$:

$$\hat{r}_t = \mathbf{z}_t^\top \boldsymbol{\gamma} + \mathbf{x}_{\text{AR},t}^\top \boldsymbol{\delta}$$

$$\min_{\boldsymbol{\gamma},\boldsymbol{\delta}} \;\frac{1}{2n}\|\mathbf{r} - \hat{\mathbf{r}}\|^2 + \lambda\!\left(\alpha \|\boldsymbol{\gamma}\|_1 + \frac{1-\alpha}{2}\|\boldsymbol{\gamma}\|^2\right)$$

### 7. Walk-forward evaluation

For test dates $t \in \{T - T_{\text{test}}, \ldots, T\}$:
- **Training set**: all observations $\tau < t$ (expanding window)
- **Forecast**: one-step-ahead $\hat{r}_t \mid \mathcal{F}_{t-1}$
- No future data ever enters any training window

---

## Macro Data Pipeline

### Sources and series

#### Core FRED (Phase 0, 5 series)
| Series | Feature | Transform | Release lag |
|---|---|---|---|
| DGS10 | `d_DGS10` | diff | 0d (daily) |
| DGS2 | `d_DGS2` | diff | 0d |
| VIXCLS | `d_VIXCLS` | diff | 0d |
| BAA10Y | `d_BAA10Y` | diff | 0d |
| DGS10–DGS2 | `d_term_spread` | diff | 0d |

#### Extended FRED (Phase 1, 13 series)

**Tier A — Daily (ffill=5)**
| Series | Feature | Transform |
|---|---|---|
| T5YIE | `d_T5YIE` | diff |
| T10YIE | `d_T10YIE` | diff |
| T5YIFR | `d_T5YIFR` | diff |
| DFF | `d_DFF` | diff |

**Tier B — Weekly (ffill=7)**
| Series | Feature | Transform | Release lag |
|---|---|---|---|
| STLFSI4 | `STLFSI4` | none | 6d |
| NFCI | `NFCI` | none | 5d |
| WALCL | `d_WALCL` | log_diff | 2d |

**Tier C — Monthly (ffill=25)**
| Series | Feature | Transform | Release lag |
|---|---|---|---|
| CPIAUCSL | `mom_CPI` | log_diff (MoM) | 15d |
| CPILFESL | `mom_CoreCPI` | log_diff (MoM) | 15d |
| PAYEMS | `d_PAYEMS` | log_diff | 5d |
| UNRATE | `d_UNRATE` | diff | 5d |
| INDPRO | `d_INDPRO` | log_diff | 16d |
| UMCSENT | `d_UMCSENT` | diff | 14d |

> **Note on CPI transforms**: YoY log-diff (12-month) was non-stationary (ADF p > 0.05) over 2012–2025 due to the 2021–2023 inflation surge. Switched to MoM log-diff (lags=1), which passes ADF at p ≈ 0 throughout the sample.

#### BLS Direct API (Phase 2, 4 series — graceful degradation)
API: BLS v2 POST (keyless: 25 req/day; registered: 500 req/day via `BLS_API_KEY`)

| BLS Series ID | Feature | Transform | Release lag |
|---|---|---|---|
| CUSR0000SAH1 | `mom_ShelterCPI` | log_diff (MoM) | 15d |
| CUSR0000SAM | `mom_MedicalCPI` | log_diff (MoM) | 15d |
| CES0500000003 | `d_HourlyEarnings` | log_diff | 5d |
| JTS00000000JOR | `d_JOLTS` | diff | 35d |

Raw data is cached to `.cache/` to avoid re-hitting the daily limit. If the limit is exceeded, BLS features are skipped gracefully — the pipeline continues with the remaining 21 series.

#### BEA NIPA (Phase 3, 3 quarterly series)
API: BEA REST API (`https://apps.bea.gov/api/data/`, key via `BEA_API_KEY`, Year=ALL)

| BEA Table | Series | Feature | Transform | Release lag |
|---|---|---|---|---|
| T10101 | A191RL | `GDP_growth_pct` | none (already %) | 30d |
| T20304 | DPCERG | `d_yoy_PCE_PI` | Δ YoY log-diff (4 qtrs) | 30d |
| T11200 | A055RC | `yoy_CorpProfits` | YoY log-diff (4 qtrs) | 45d |

> **Note on PCE transform**: YoY log-diff and QoQ log-diff of DPCERG are non-stationary over 2010–2025 (ADF p > 0.05). The first difference of the YoY log-diff (`d_yoy_log_diff` transform = `log().diff(4).diff()`) is stationary at ADF p = 0.039 (post-2010). This represents the *change in annual PCE inflation* — economically meaningful and stationary. Forward-fill limit = 70 trading days (one quarter).

### Stationarity transforms

| Transform key | Operation | Typical use |
|---|---|---|
| `diff` | $x_t - x_{t-\ell}$ | Yield/rate levels |
| `log_diff` | $\log x_t - \log x_{t-\ell}$ | Price/activity indices |
| `yoy_log_diff` | $\log x_t - \log x_{t-12}$ (monthly) | Money supply, GDP level |
| `d_yoy_log_diff` | $(\log x_t - \log x_{t-4}) - (\log x_{t-1} - \log x_{t-5})$ | Price indices with structural breaks |
| `pct_change` | $(x_t - x_{t-\ell}) / x_{t-\ell}$ | Generic % change |
| `none` | $x_t$ | Already-stationary composites (NFCI, STLFSI4) |

### Look-ahead prevention

Every monthly/quarterly series carries a `release_lag` (calendar days). After the stationarity transform, the series index is shifted forward:

$$\tilde{t} = t^{\text{ref}} + \ell_k$$

Then aligned to the daily trading calendar via **union-index forward-fill**:

```
union_idx = trading_idx ∪ transformed_idx  (sorted)
aligned   = transformed.reindex(union_idx).ffill(limit=δ_k).reindex(trading_idx)
```

This two-step reindex ensures that publication dates landing on weekends/holidays still propagate correctly to the next trading day, with the ffill limit preventing stale data from propagating too far.

---

## Project Structure

```
mmforecasting/
├── notebooks/
│   └── forecasting_analysis.ipynb   # Main analysis (101 cells)
├── src/
│   ├── macro_utils.py               # Shared: index normalisation, transforms, validation
│   ├── fred_pipeline.py             # Core FRED (5) + Extended FRED (13 series)
│   ├── bls_pipeline.py              # BLS direct API (4 series, graceful degradation)
│   ├── bea_pipeline.py              # BEA NIPA API (3 quarterly series)
│   ├── arimax_models.py             # PCA+EN and PLS implementations
│   └── evaluation.py               # Walk-forward metrics
├── .cache/                          # Parquet cache (gitignored)
├── .env                             # API keys (FRED, BEA, BLS, Alpaca, News)
├── requirements.txt
└── README.md
```

### `src/macro_utils.py` — shared contract

All pipelines import from here to guarantee a consistent index contract:

- `_to_date_index(idx)` — tz-aware or tz-naive → tz-naive midnight `DatetimeIndex`
- `apply_release_lag(df, lag_days)` — shifts index forward, re-normalises to midnight
- `apply_transform(series, transform, lags)` — dispatch table for all transform types
- `fetch_chunked(fetch_fn, start, end, chunk_years)` — splits API calls by year window
- `validate_features(df, label, ...)` — non-NaN fraction + ADF gate per column

---

## Quick Start

```bash
# Activate environment
source .venv/bin/activate

# Copy and fill API keys
cp .env.example .env   # add FRED_API_KEY, BEA_API_KEY (BLS key optional)

# Launch notebook
jupyter notebook notebooks/forecasting_analysis.ipynb
```

### Configuration

```python
TICKER     = "AAPL"         # Target equity
START_DATE = "2012-01-01"   # Data start (FRED extended data available from ~2004)
END_DATE   = "2025-12-31"   # Data end
TEST_SIZE  = 252            # Walk-forward test window (trading days)
N_AR_LAGS  = 5              # AR lags in mean equation
N_EXOG_LAGS = 5             # Macro feature lags
PCA_K      = 10             # PCA components fed to ElasticNet
N_COMP_PLS = 5              # PLS latent components
```

---

## Diagnostic Gates

Every data source is validated before downstream use:

1. **Non-NaN fraction** ≥ 50% per column (expected ≥ 95% for post-2010 data)
2. **ADF stationarity** p < 0.05 for all transformed columns
3. **Release-lag lookahead check**: no value from period $t$ appears before $t + \ell_k$

The design matrix cell prints a full summary upon completion:

```
Macro sources        : 3 block(s)
Total macro vars     : 21
Aligned observations : 3,479
Exog features        : 105  (21 macro vars × 5 lags)
```

---

## Interpretation Notes

- This project evaluates **statistical predictability**, not economic profitability. No transaction costs, slippage, or risk constraints are applied.
- Residual dependence may remain due to regime shifts, structural breaks, or long-memory effects.
- Results should be interpreted as **model comparison**, not trading signals.

---

## License

MIT
