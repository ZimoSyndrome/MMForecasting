"""
Restructure forecasting_analysis.ipynb into a clean quantitative research pipeline:
  Data Ingestion → EDA → Feature Engineering → Model Specification
  → Walk-Forward Backtesting → Results & Comparison → Conclusions

Cells dropped: orig[39] (old A.6 header), orig[46] (old Part B banner), orig[63] (old Part D banner)
New markdown cells inserted: 15 section/sub-section headers
New code cells inserted: 2 (ETF correlation heatmap, FRED macro visualization)
"""

import json, copy

NB = "/Users/zimo/mmforecasting/notebooks/forecasting_analysis.ipynb"

with open(NB) as f:
    nb = json.load(f)

orig = nb["cells"]  # original 84 cells, indexed 0..83

def take(i):
    return copy.deepcopy(orig[i])

def rewrite(i, new_src):
    c = copy.deepcopy(orig[i])
    c["source"] = new_src
    return c

def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src}

def code(src):
    return {"cell_type": "code", "execution_count": None,
            "metadata": {}, "outputs": [], "source": src}

# ══════════════════════════════════════════════════════════════════════════════
# BUILD NEW CELL LIST
# ══════════════════════════════════════════════════════════════════════════════
new_cells = []

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 0: Project Overview & Setup  (new idx 0–7)
# ─────────────────────────────────────────────────────────────────────────────
new_cells.append(rewrite(0, """\
# Equity Return Forecasting: A Multi-Model Risk Engine

**A systematic walk-forward backtesting study comparing five model families — classical \
time-series, gradient boosting, deep learning, and macro-enhanced ARX-GARCH — for \
next-day equity return prediction and probabilistic risk quantification.**

---

## Project at a Glance

| Dimension | Detail |
|---|---|
| **Objective** | Forecast next-day log return; estimate conditional risk (μ, σ, VaR) |
| **Evaluation** | Expanding-window walk-forward backtest — zero lookahead bias |
| **Models compared** | ARIMA-GARCH · XGBoost · LSTM · ARX+PCA+EN · ARX+PLS (+ FRED macro variants) |
| **Exogenous data** | 11 SPDR sector ETFs + 5 FRED macro-financial daily change series |
| **Probabilistic outputs** | μ · σ² · 95% CI · VaR₉₅ · VaR₉₉ (ARX-GARCH models) |

## Central Research Question

> *Does cross-sectional sector information and macro-financial change variables improve \
> next-day return forecasting beyond a pure ARIMA-GARCH baseline — and if so, does the \
> improvement appear in point accuracy, directional edge, or risk calibration?*

## Pipeline

```
1. Data Ingestion  →  2. EDA  →  3. Feature Engineering  →  4. Model Specification
→  5. Walk-Forward Backtesting  →  6. Results & Comparison  →  7. Conclusions
```\
"""))


new_cells.append(rewrite(1, """\
## Model Inventory

| # | Model | Family | Mean Equation | Variance | Exogenous? | Outputs |
|---|---|---|---|---|---|---|
| 1 | ARIMA-GARCH | Statistical | ARIMA(p,0,q) — AIC grid | GARCH(1,1) | ✗ | Point |
| 2 | XGBoost | Gradient Boosting | AR lags + rolling vol | None | ✗ | Point |
| 3 | LSTM | Neural Network | Recurrent sequence | None | ✗ | Point |
| 4 | ARX + PCA + EN | ARX-GARCH | PCA(k=5) + ElasticNet | GARCH(1,1) | ✓ ETF | Full ★ |
| 5 | ARX + PLS | ARX-GARCH | PLS(c=3) + OLS | GARCH(1,1) | ✓ ETF | Full ★ |
| 6 | ARX + PCA + EN + FRED | ARX-GARCH | PCA(k=5)+EN on ETF+macro | GARCH(1,1) | ✓ ETF+FRED | Full ★ |
| 7 | ARX + PLS + FRED | ARX-GARCH | PLS(c=3) on ETF+macro | GARCH(1,1) | ✓ ETF+FRED | Full ★ |

★ *Full outputs: predicted mean, variance, std, 95% confidence interval, VaR₉₅, VaR₉₉*

**Evaluation metrics:** MSE · MAE · Direction Accuracy · CI Coverage (95%) · VaR Breach Rate (95%)

---

## Table of Contents
1. [Data Ingestion](#ingestion)
2. [Exploratory Data Analysis](#eda)
3. [Feature Engineering](#features)
4. [Model Specification & Hyperparameter Selection](#model-spec)
5. [Walk-Forward Backtesting](#backtest)
6. [Results & Comparison](#results)
7. [Conclusions & Limitations](#conclusions)\
"""))

new_cells.append(take(2))   # Setup anchor
new_cells.append(take(3))   # Global config (TICKER, START_DATE, TEST_SIZE, …)
new_cells.append(take(10))  # Exog config (PCA_K, N_EXOG_LAGS, …) — pure constants, belongs in setup
new_cells.append(take(4))   # pip install + all main imports
new_cells.append(take(5))   # sklearn ARIMAX imports
new_cells.append(take(7))   # fetch_data_alpaca / fetch_data function definitions


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: Data Ingestion  (new idx 8–16)
# ─────────────────────────────────────────────────────────────────────────────
new_cells.append(md("""\
<a id='ingestion'></a>

---
## 1. Data Ingestion

All raw data is sourced and loaded before any transformation or model fitting.
Three sources feed the pipeline:

| # | Source | Provider | What is fetched |
|---|---|---|---|
| 1 | Target equity | Alpaca API / yfinance fallback | Adjusted daily close → log return |
| 2 | Sector ETFs | yfinance | 11 SPDR sector ETF adjusted daily close → log return |
| 3 | Macro-financial | FRED (pandas-datareader) | 10Y/2Y yields, term spread, VIX, credit spread |

**No-lookahead guarantee:** exogenous series enter models only through lagged copies (lag ≥ 1).
FRED series are forward-filled over weekends/holidays (max 5 days) using only information
available on or before the trading date — never future values.\
"""))

new_cells.append(rewrite(6, """\
### 1.1 Target Equity Data

Daily adjusted close prices are downloaded for the configured `TICKER`.
Log returns are computed as *r_t = ln(P_t / P_{t−1})*.

**Why log returns?**
- **Stationarity:** prices are I(1); log-returns are approximately I(0) ✓
- **Additivity:** multi-day log-returns sum correctly over time
- **Scale-invariance:** a 1% return has the same meaning regardless of price level\
"""))

new_cells.append(take(8))   # df = fetch_data(…); df['Return'] = …

new_cells.append(md("### 1.2 Sector ETF Exogenous Data"))

new_cells.append(rewrite(9, """\
Eleven SPDR sector ETFs are used as cross-sectional exogenous regressors — their lagged
daily returns proxy industry momentum and rotation effects that the target's own history
cannot capture.

**Self-regression rule:**
- Target = SPY (broad market) → exclude SPY; include all 11 sector ETFs
- Target = sector ETF (e.g. XLK) → exclude that ETF; include SPY + remaining 10
- Target = individual stock → all 12 tickers eligible

Each included ticker contributes 5 lagged daily return features (lags 1–5) to the
exogenous design matrix, for a total of up to 55 features before dimensionality reduction.\
"""))

new_cells.append(take(11))  # fetch exog_returns via yfinance

new_cells.append(md("### 1.3 FRED Macro-Financial Data"))

new_cells.append(rewrite(74, """\
Five macro-financial series are downloaded from the Federal Reserve Economic Data (FRED) database.
**First differences (changes) are used** — not raw levels — to ensure stationarity and to
isolate daily *shocks* rather than persistent level effects.

| FRED Code | Description | Transform used |
|---|---|---|
| DGS10 | 10-Year Treasury Constant Maturity Yield | Δ daily change |
| DGS2 | 2-Year Treasury Constant Maturity Yield | Δ daily change |
| DGS10 − DGS2 | Term Spread (computed from above) | Δ change of spread |
| VIXCLS | CBOE VIX Index | Δ daily change |
| BAA10Y | Moody's BAA Corporate − 10Y Treasury Spread | Δ daily change |

**Rationale for changes over levels:**
Yield *levels* are I(1) non-stationary. First-differencing produces stationary shocks:
a sudden VIX spike or credit spread widening on day *t* is the relevant signal for
next-day returns — not the prevailing level itself.\
"""))

new_cells.append(take(75))  # fetch fred_features via fred_pipeline


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: Exploratory Data Analysis  (new idx 17–28)
# ─────────────────────────────────────────────────────────────────────────────
new_cells.append(md("""\
<a id='eda'></a>

---
## 2. Exploratory Data Analysis

Before fitting any model, the statistical properties of all data are examined to
motivate the modelling choices made in Parts 3 and 4.

1. **Return distribution & volatility clustering** — visual inspection for fat tails and ARCH effects
2. **Sector ETF cross-correlations** — motivates PCA compression of the 55-feature exog block
3. **FRED macro change series** — confirms stationarity and identifies dominant shock regimes
4. **Stationarity (ADF tests)** — determines integration order *d* for ARIMA, confirms I(0)
5. **ARIMA-GARCH framework motivation** — two-step estimation rationale\
"""))

new_cells.append(take(14))  # Price, log-price, log-return visualization

# NEW: ETF correlation heatmap
new_cells.append(code("""\
# ── Sector ETF return correlation matrix ─────────────────────────────────────
etf_corr = exog_returns.dropna().corr()

fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(etf_corr.values, cmap="RdYlGn", vmin=-0.4, vmax=1.0, aspect="auto")
plt.colorbar(im, ax=ax, shrink=0.75, label="Pearson correlation")

tickers_etf = list(etf_corr.columns)
ax.set_xticks(range(len(tickers_etf)))
ax.set_yticks(range(len(tickers_etf)))
ax.set_xticklabels(tickers_etf, rotation=45, ha="right", fontsize=9)
ax.set_yticklabels(tickers_etf, fontsize=9)

for i in range(len(tickers_etf)):
    for j in range(len(tickers_etf)):
        val = etf_corr.values[i, j]
        ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                fontsize=7, color="white" if val > 0.8 else "black")

ax.set_title(
    f"Sector ETF Return Correlation Matrix\\n"
    f"({exog_returns.index[0].date()} – {exog_returns.index[-1].date()}, "
    f"{len(exog_returns)} trading days)",
    fontsize=11
)
plt.tight_layout()
plt.show()

_corr_lo = np.tril_indices_from(etf_corr.values, k=-1)
_corr_vals = etf_corr.values[_corr_lo]
print(f"Average pairwise correlation : {_corr_vals.mean():.3f}")
print(f"Min / Max                    : {_corr_vals.min():.3f} / {_corr_vals.max():.3f}")
print()
print("High pairwise correlations motivate PCA — the 55-feature ETF space is highly collinear.")
print("PCA(k=5) compresses this into 5 decorrelated components before Elastic Net selection.")\
"""))

# NEW: FRED macro visualization
new_cells.append(code("""\
# ── FRED macro-financial variables: daily change series ───────────────────────
_fred_labels = {
    "d_DGS10":       "Δ 10Y Treasury Yield",
    "d_DGS2":        "Δ 2Y Treasury Yield",
    "d_term_spread": "Δ Term Spread (10Y−2Y)",
    "d_VIXCLS":      "Δ VIX",
    "d_BAA10Y":      "Δ Credit Spread (BAA−10Y)",
}

fig, axes = plt.subplots(3, 2, figsize=(14, 9))
axes_flat = axes.flatten()

for j, (col, label) in enumerate(_fred_labels.items()):
    ax = axes_flat[j]
    series = fred_features[col].dropna()
    ax.plot(series.index, series.values, lw=0.7, color="steelblue", alpha=0.85)
    ax.axhline(0, color="black", ls="--", lw=0.5, alpha=0.5)
    ax.fill_between(series.index, series.values, 0,
                    where=(series.values > 0), alpha=0.12, color="green")
    ax.fill_between(series.index, series.values, 0,
                    where=(series.values < 0), alpha=0.12, color="red")
    ax.set_title(label, fontsize=9)
    ax.set_ylabel("Daily Change", fontsize=7)
    ax.tick_params(labelsize=7)

axes_flat[-1].set_visible(False)
plt.suptitle(
    "FRED Macro-Financial Variables — First-Difference (Change) Series\\n"
    "All 5 series enter ARX-GARCH models as lagged exogenous features (lags 1–5)",
    fontsize=11
)
plt.tight_layout()
plt.show()

print("FRED macro change series — summary statistics (aligned modelling period):")
print(fred_features[list(_fred_labels.keys())].describe().round(5))\
"""))

new_cells.append(rewrite(15, """\
### 2.1 ARIMA-GARCH: Two-Step Framework Motivation

Financial returns exhibit two well-documented empirical regularities:

1. **Weak mean dependence** — small, time-varying autocorrelation in returns (ARIMA models this)
2. **Volatility clustering** — large absolute moves follow large absolute moves (GARCH models this)

**Two-step estimation:**
> Step 1 — Mean model: Fit ARIMA(p,0,q) on log-returns; select p, q by AIC grid search.
> Extract residuals ε_t = r_t − ARIMA_forecast_t
>
> Step 2 — Variance model: Fit GARCH(1,1) on {ε_t}.
> Conditional variance: σ²_t = ω + α·ε²_{t−1} + β·σ²_{t−1}

This separation is valid when mean and variance dynamics are orthogonal — confirmed by
ARCH-LM tests on ARIMA residuals (see Section 4.3). The resulting (μ̂_t, σ̂_t) pair is
then used to construct confidence intervals and VaR estimates in the walk-forward backtest.\
"""))

new_cells.append(rewrite(16, """\
### 2.2 Stationarity Tests (Augmented Dickey-Fuller)

The integration order *d* determines whether differencing is required before ARIMA modelling.
ADF tests are applied to raw price, log-price, and log-returns.

| Series | H₀ | Decision |
|---|---|---|
| Price | Unit root (I(1)) | Expected: *fail to reject* at 5% |
| Log-price | Unit root (I(1)) | Expected: *fail to reject* at 5% |
| Log-return | Stationary (I(0)) | Expected: *reject* at 5% ✓ |

**Implication:** ARIMA is fitted directly on log-returns with d = 0.\
"""))

new_cells.append(take(17))  # adf_test + find_best_d_by_adf functions
new_cells.append(take(18))  # Run ADF on price / log-price / log-returns
new_cells.append(take(19))  # "Choice of Modelling Series" markdown
new_cells.append(take(20))  # modelling_series selection code

new_cells.append(rewrite(76, """\
### 2.3 FRED Feature Construction Details

**Forward-fill rule (no look-ahead):**
FRED publishes yield and VIX data on US business days only. On weekends and Fed holidays,
the most recently available *change* is carried forward up to 5 calendar days — this assumes
"no new macro news" rather than interpolating. All forward-filling uses only data available
on or before the trading date.

**Lagging:** each of the 5 FRED change series is lagged 1–5 days (identical to ETF lags),
expanding the exogenous block from 55 → 80 features:

| Block | Columns | After lagging |
|---|---|---|
| Sector ETF returns (11 ETFs) | 11 | 55 (× 5 lags) |
| FRED macro changes (5 series) | 5 | 25 (× 5 lags) |
| **Combined exogenous block** | **16** | **80** |

PCA and PLS are refitted on the full 80-column block at each walk-forward step,
so the macro variables compete with sector ETF features on equal footing.\
"""))

new_cells.append(take(12))  # build_design_matrix function + call → y_dm, X_ar_dm, X_exog_dm
new_cells.append(take(13))  # ETF construction methodology markdown


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: Feature Engineering  (new idx 29–39)
# ─────────────────────────────────────────────────────────────────────────────
new_cells.append(md("""\
<a id='features'></a>

---
## 3. Feature Engineering

Three distinct feature sets are constructed for the respective model families.
All features use strictly past information — same-day data never enters any feature.

| Feature Set | Model(s) | Columns |
|---|---|---|
| ARX exogenous lags (ETF-only) | PCA+EN, PLS | 55 (11 ETFs × 5 lags) + 5 AR lags |
| ARX exogenous lags (ETF + FRED) | PCA+EN+FRED, PLS+FRED | 80 (16 series × 5 lags) + 5 AR lags |
| Tabular lags + rolling volatility | XGBoost | 6 (5 AR lags + 21-day realised vol) |
| Sliding-window sequences | LSTM | seq_len × 1 |

**Dimensionality reduction** (PCA+EN and PLS) is applied within each walk-forward step,
fitted on the training split only and applied to the test row. No information from
the test window enters the compression stage.\
"""))

new_cells.append(md("""\
### 3.1 ARX-GARCH Exogenous Matrix — ETF-Only

`build_design_matrix` (defined in Section 2.3 above) produced `y_dm`, `X_ar_dm`, and
`X_exog_dm` — the aligned (y, AR lags, exog lags) triplet used by the ETF-only ARX-GARCH models.

Key properties:
- All ETF lags computed as `shift(k)` for k = 1, …, N_EXOG_LAGS — strictly no same-day data
- `dropna()` ensures full alignment; effective start shifts to June 2018 due to XLC launch date
- AR lags of target are separated from exog lags: PCA+EN can select or discard AR lags via Elastic Net\
"""))

new_cells.append(rewrite(45, """\
### 3.2 XGBoost Feature Set

XGBoost operates on a tabular feature matrix: AR lags 1–5 of the target return and a
21-day rolling realised volatility proxy (std of recent returns × √252 for annualisation).

**Walk-forward discipline:** the model is retrained from scratch at each step on the
expanding training window. Hyperparameters are fixed (literature-defaults: n_estimators=200,
max_depth=4, learning_rate=0.05) — no within-window tuning is applied.

**Why include rolling volatility?**
Equity returns exhibit **volatility feedback**: high-vol periods tend to persist.
Including σ_{t−1} gives XGBoost a non-linear proxy for the variance regime — something
that helps tree models capture GARCH-like behaviour without explicit variance modelling.\
"""))

new_cells.append(take(47))  # feature_df, X_full, y_full construction

new_cells.append(rewrite(50, """\
### 3.3 LSTM Architecture & Sequence Construction

A single-layer LSTM with a linear output head is trained to predict r_{t+1} from a
sliding window of recent returns. The network architecture is defined below.

**Design decisions:**
- **One LSTM layer** — simpler than stacked LSTM; sufficient for low-dimensional return sequences
- **Linear output** — regression task (not classification of sign)
- **Adam optimiser** — adaptive learning rate; standard choice for financial sequence modelling
- **No dropout** — sequences are short; dropout often hurts rather than helps on financial returns

**Limitation:** daily return sequences contain very low signal-to-noise (SNR ≪ 1).
LSTM frequently overfits on financial data without aggressive regularisation — its
out-of-sample performance should be interpreted cautiously.\
"""))

new_cells.append(take(51))  # LSTMNet class + LSTM_CONFIG
new_cells.append(take(52))  # make_sequences function

new_cells.append(md("""\
### 3.4 Macro-Enhanced ARX Design Matrix (ETF + FRED)

The FRED macro change series (from Section 1.3) are appended to the ETF exogenous block,
creating the 80-column design matrix used by the +FRED model variants.
PCA and PLS are independently refitted on this expanded matrix at each walk-forward step.\
"""))

new_cells.append(take(77))  # build combined exog + y_dm_m, X_exog_macro_dm


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: Model Specification & Hyperparameter Selection  (new idx 40–62)
# ─────────────────────────────────────────────────────────────────────────────
new_cells.append(md("""\
<a id='model-spec'></a>

---
## 4. Model Specification & Hyperparameter Selection

All hyperparameter choices are made using **in-sample data only** — no test-period
observations are consulted at this stage.

| Model | Hyperparameter Selection | Criterion |
|---|---|---|
| ARIMA(p,0,q) | Grid search p, q ∈ {0,…,3} | Minimum AIC |
| GARCH(1,1) | Distribution grid search (Normal / t / Skewed-t) | Minimum AIC |
| XGBoost | Fixed configuration | Literature defaults |
| LSTM | Fixed architecture | Literature defaults |
| PCA + EN | k = 5 PCA components; λ via ElasticNetCV | 3-fold CV MSE per walk-forward step |
| PLS | c = 3 latent components | In-sample fit per walk-forward step |

Walk-forward models (XGBoost, LSTM, PCA+EN, PLS) are refitted from scratch at every
prediction step — **parameters change, hyperparameters do not**.\
"""))

new_cells.append(rewrite(21, """\
### 4.1 ARIMA(p,0,q) Grid Search — Mean Model

Log-returns are already I(0) (confirmed in Section 2.2), so only p and q are searched (d = 0).

**Grid:** p ∈ {0, 1, 2, 3}, q ∈ {0, 1, 2, 3} — 16 candidate models
**Criterion:** Akaike Information Criterion (AIC) = −2 log L + 2k — penalises complexity
**Implementation:** `statsmodels.tsa.arima.model.ARIMA` with MLE estimation

AIC balances in-sample fit against parameter count, preventing overfitting to the
training set. It is preferred over BIC here because the financial time series is long
enough that the stronger BIC penalty is not necessary.\
"""))

new_cells.append(take(22))  # fit_arima function
new_cells.append(take(23))  # ARIMA grid search loop → arima_df
new_cells.append(take(24))  # select best ARIMA → BEST_P, BEST_Q, best_arima_fit
new_cells.append(take(25))  # best_arima_fit.summary()

new_cells.append(rewrite(26, """\
### 4.2 GARCH(1,1) Grid Search — Variance Model

The ARCH-LM test above confirms significant conditional heteroskedasticity in ARIMA residuals.
GARCH(1,1) is estimated on scaled residuals to avoid numerical issues.

**Model:** σ²_t = ω + α·ε²_{t−1} + β·σ²_{t−1}
**Stationarity constraint:** α + β < 1 (mean-reverting conditional variance)

**Why search over innovation distributions?**
Daily equity returns exhibit fat tails — the Normal distribution underestimates
the probability of extreme moves by 2–5×. Student-t and Skewed Student-t allow heavier
tails and potential asymmetry. The best distribution is selected by AIC.

**`rescale=False` note:** the `arch` library defaults to rescaling the series by ×100
internally. Using `rescale=False` ensures `forecast().variance` is returned in
original-scale units, preventing a ×10,000 inflation of σ² in walk-forward forecasts.\
"""))

new_cells.append(take(27))  # extract ARIMA residuals + scale
new_cells.append(take(28))  # ARCH-LM test on ARIMA residuals (pre-GARCH)
new_cells.append(take(29))  # GARCH grid search → garch_df
new_cells.append(take(30))  # select best GARCH → BEST_DIST, best_garch_fit
new_cells.append(take(31))  # best_garch_fit.summary()

new_cells.append(rewrite(32, """\
### 4.3 GARCH Residual Diagnostics

Two specification tests on GARCH standardized residuals z_t = ε_t / σ_t:

| Test | H₀ | Rejection means |
|---|---|---|
| Ljung-Box | No autocorrelation in z_t | Mean model (ARIMA) is incomplete |
| ARCH-LM | No ARCH effects in z²_t | Variance model (GARCH) is incomplete |

A well-specified GARCH model **fails to reject** both null hypotheses at 5%, confirming
that the two-step procedure has successfully modelled both mean and variance dynamics.
Residual plots (standardized residuals, ACF, Q-Q) provide visual confirmation.\
"""))

new_cells.append(take(33))  # Ljung-Box test
new_cells.append(take(34))  # "Why autocorrelation persists" methodology markdown
new_cells.append(take(35))  # ARCH-LM test on GARCH std residuals
new_cells.append(take(36))  # 2×2 diagnostic plots

new_cells.append(rewrite(37, """\
### 4.4 In-Sample Conditional Volatility

The fitted GARCH(1,1) conditional volatility σ_t is overlaid on the ARIMA residuals
to confirm that GARCH is tracking volatility clusters. Periods of high |ε_t| should
co-move with high σ_t — the key empirical justification for GARCH modelling.

Annualised conditional volatility = σ_t × √252.\
"""))

new_cells.append(take(38))  # in-sample vol bands code

new_cells.append(rewrite(55, """\
### 4.5 ARX-GARCH Configuration

ARIMA-GARCH assumes the only predictive information is in the target's own history.
The ARX-GARCH extension tests whether **cross-sectional exogenous information** — sector
ETF momentum and macro-financial shocks — improves the mean forecast.

**Same two-step structure as ARIMA-GARCH**, but the mean equation is replaced by a
regularised regression on [target AR lags + compressed exogenous features]:

```
Mean:     μ̂_t = EN or OLS regression (AR lags + PCA/PLS factors)
Residuals: ε_t = r_t − μ̂_t
Variance:  σ²_t = GARCH(1,1) on {ε_t}   ← same as before
```

**ARX vs ARIMAX distinction:**
The current implementation is an *ARX-GARCH* (no MA terms in mean equation).
True *ARIMAX-GARCH* would use `statsmodels ARIMA(endog, exog=PCA_factors)` to jointly
estimate AR, MA, and exogenous coefficients by MLE, then GARCH on ARIMAX residuals.
Adding MA terms is a natural extension; for short-memory equity returns the practical
difference is small.\
"""))

new_cells.append(take(56))  # DIST_ARIMAX, _arimax_forecast_record, _fit_arimax_garch_var

new_cells.append(rewrite(57, """\
### 4.6 ARX-GARCH | PCA + Elastic Net — Methodology

**The dimensionality challenge:** 80 lagged exogenous features (or 55 for ETF-only) are
highly correlated. Feeding them directly into OLS or ARIMA would overfit and produce
unstable coefficients. Two-stage compression solves this:

**Stage 1 — Unsupervised compression (PCA):**
Principal Component Analysis extracts k=5 orthogonal components from the exogenous block,
ordered by explained variance. Fitted on the training split only; applied to the test row
using training-period loadings. This decorrelates the feature space without using any y information.

**Stage 2 — Sparse selection (Elastic Net):**
Candidate matrix = [AR lags 1–5, PC1–PC5]. ElasticNetCV (l1_ratio grid, 3-fold CV)
selects a sparse linear combination. The model name reflects which AR lags and PC indices
were selected in >50% of walk-forward steps.

**Stage 3 — Variance (GARCH):**
GARCH(1,1) on the regression residuals. `rescale=False` used throughout.\
"""))

new_cells.append(rewrite(60, """\
### 4.7 ARX-GARCH | PLS — Methodology

**The key difference from PCA+EN:** PLS extracts latent components that **directly maximise
covariance with y** (supervised), rather than maximising variance of X (unsupervised).

**Trade-offs vs PCA+EN:**

| Dimension | PCA + EN | PLS |
|---|---|---|
| Compression | Unsupervised (max X-variance) | Supervised (max X-y covariance) |
| Feature selection | Elastic Net (sparse, data-driven) | Fixed c=3 components; OLS |
| Noise robustness | High (PCA is agnostic to y) | Lower (components tuned to y) |
| Interpretability | PC loadings show ETF groupings | PLS loadings show return-relevant ETFs |
| Overfitting risk | Lower (regularised) | Higher on small training windows |

**Mean equation:** OLS on [fixed AR lags 1–N_AR_FIXED, PLS X-scores].
**Variance:** GARCH(1,1) on OLS residuals, identical to PCA+EN.\
"""))


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: Walk-Forward Backtesting  (new idx 63–88)
# ─────────────────────────────────────────────────────────────────────────────
new_cells.append(md("""\
<a id='backtest'></a>

---
## 5. Walk-Forward Backtesting

**Protocol:** Expanding-window evaluation over the final `TEST_SIZE` trading days.

```
Step t=0 : Train on [start … T−TEST_SIZE−1],  predict day T−TEST_SIZE
Step t=1 : Train on [start … T−TEST_SIZE],    predict day T−TEST_SIZE+1
        ⋮
Step t=k : Train on [start … T−2],            predict day T−1
```

This protocol:
- **Eliminates lookahead bias** — no future return enters any training window
- **Simulates live trading** — each step mirrors the information available at close-of-day t
- **Stress-tests stability** — models that degrade across the test window are visible in rolling metrics

ARX-GARCH models produce full **probabilistic** outputs at each step: (μ̂_t, σ̂_t) →
95% CI and VaR₉₅/VaR₉₉. XGBoost and LSTM produce point forecasts only.\
"""))

# ── 5.1 ARIMA-GARCH ──
new_cells.append(md("### 5.1 ARIMA-GARCH Baseline Backtest"))
# orig[39] is DROPPED — its "### A.6" header is replaced by the line above
new_cells.append(take(40))  # ARIMA-GARCH backtest loop
new_cells.append(take(41))  # store arima_results DataFrame
new_cells.append(take(42))  # z-score arrays + backtest_threshold function
new_cells.append(take(43))  # threshold sweep methodology markdown
new_cells.append(take(44))  # threshold sweep code

# ── 5.2 XGBoost ──
new_cells.append(md("### 5.2 XGBoost Walk-Forward Backtest"))
# orig[46] is DROPPED — redundant Part B banner
new_cells.append(take(48))  # XGBoost backtest loop
new_cells.append(take(49))  # store xgb_results

# ── 5.3 LSTM ──
new_cells.append(md("### 5.3 LSTM Walk-Forward Backtest"))
new_cells.append(take(53))  # LSTM backtest loop (intentionally reassigns y_full/X_full)
new_cells.append(take(54))  # store lstm_results

# ── 5.4 PCA+EN (ETF) ──
new_cells.append(md("### 5.4 ARX-GARCH | PCA + Elastic Net Backtest  (ETF Features)"))
new_cells.append(take(58))  # PCA+EN backtest loop → pca_en_results, PCA_EN_NAME
new_cells.append(take(59))  # PCA+EN metrics print + full diagnostic report

# ── 5.5 PLS (ETF) ──
new_cells.append(md("### 5.5 ARX-GARCH | PLS Backtest  (ETF Features)"))
new_cells.append(take(61))  # PLS backtest loop → pls_results, PLS_NAME
new_cells.append(take(62))  # PLS metrics print + full diagnostic report

# ── 5.6 PCA+EN+FRED ──
new_cells.append(md("### 5.6 ARX-GARCH | PCA + Elastic Net Backtest  (ETF + FRED Features)"))
new_cells.append(take(78))  # PCA+EN+FRED backtest loop → pca_en_mac_results
new_cells.append(take(79))  # PCA+EN+FRED metrics + diagnostics

# ── 5.7 PLS+FRED ──
new_cells.append(md("### 5.7 ARX-GARCH | PLS Backtest  (ETF + FRED Features)"))
new_cells.append(take(80))  # PLS+FRED backtest loop → pls_mac_results
new_cells.append(take(81))  # PLS+FRED metrics + diagnostics


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6: Results & Comparison  (new idx 89–106)
# ─────────────────────────────────────────────────────────────────────────────
new_cells.append(md("""\
<a id='results'></a>

---
## 6. Results, Diagnostics & Model Comparison

All seven models are evaluated on the identical held-out test window.

**Metrics:**

| Metric | Models | Formula | Target |
|---|---|---|---|
| MSE | All | E[(y − μ̂)²] | ↓ lower is better |
| MAE | All | E[|y − μ̂|] | ↓ lower is better |
| Direction Accuracy | All | P(sign(y) = sign(μ̂)) | ↑ >52% is meaningful |
| CI Coverage (95%) | ARX-GARCH only | P(μ̂−1.96σ̂ ≤ y ≤ μ̂+1.96σ̂) | ≈ 95% |
| VaR Breach Rate (95%) | ARX-GARCH only | P(y < −VaR₉₅) where VaR₉₅ = −μ̂+1.645σ̂ | ≈ 5% |

**Why direction accuracy matters more than MSE for trading:**
Returns are close to white noise — MSE differences between models are tiny (4th decimal place)
and may not be statistically distinguishable. Direction accuracy is more actionable:
each percentage point above 50% directly translates to a trading edge when compounded over
many observations.\
"""))

# ── 6.1 Baseline Comparison ──
new_cells.append(md("### 6.1 Baseline Model Metrics — ARIMA-GARCH, XGBoost, LSTM"))
# orig[63] DROPPED — old Part D anchor, replaced by section header above
new_cells.append(take(64))  # build comparison DataFrame
new_cells.append(take(65))  # compute_metrics function + per-model metrics_df
new_cells.append(take(66))  # time-series forecast plots — 3 models
new_cells.append(take(67))  # bar-chart metric comparison — 3 models

# ── 6.2 Extended Comparison (all 5 models with probabilistic metrics) ──
new_cells.append(rewrite(70, """\
### 6.2 Extended Metrics — Five Models (Including ARX-GARCH Probabilistic Outputs)

The two ARX-GARCH models produce full probabilistic forecasts, enabling two additional
risk calibration metrics beyond the baseline comparison:

**CI Coverage (95%):** what fraction of realized returns fall inside the predicted 95%
confidence interval [μ̂ − 1.96σ̂,  μ̂ + 1.96σ̂]?
- Target: ≈ 95%.  Too high → overly wide (conservative) bands.  Too low → underestimates risk.

**VaR Breach Rate (95%):** what fraction of realized returns breach the predicted VaR threshold?
(breach: actual return < −VaR₉₅ = μ̂ − 1.645σ̂)
- Target: ≈ 5%.  Too high → VaR is systematically too optimistic (dangerous for risk management).

Well-calibrated probabilistic forecasts enable dynamic position sizing, stop-loss setting,
and regulatory capital calculations (Basel III IMA backtesting requirements).\
"""))

new_cells.append(take(71))  # compute_full_metrics + 5-model report
new_cells.append(take(72))  # ARIMAX CI-band forecast visualization
new_cells.append(take(73))  # predicted volatility comparison plot

# ── 6.3 Macro Extension Comparison ──
new_cells.append(md("""\
### 6.3 FRED Macro Extension — ETF-Only vs ETF + FRED

Controlled comparison: the only change between the two model variants is the exogenous block.
All other settings (walk-forward protocol, PCA k, PLS c, EN hyperparameters, GARCH distribution)
are held constant.

**What the delta rows show:**
- **ΔMSE < 0** → macro variables improve point accuracy
- **ΔDir Acc > 0** → macro variables improve directional edge
- **ΔCI Coverage closer to 95%** → risk calibration improved
- **ΔVaR Breach closer to 5%** → VaR estimates more accurate

The most likely improvement channel is **ΔVIX** — sudden risk-aversion shocks on day t
carry directional information about day t+1 not contained in lagged sector ETF returns.\
"""))

new_cells.append(take(82))  # macro comparison report

# ── 6.4 Final Summary ──
new_cells.append(md("### 6.4 Final Summary Report"))
new_cells.append(rewrite(68, """\
The table below ranks all models and identifies winners on each metric.
Models are compared on their test-window performance only — in-sample fit is not reported.\
"""))
new_cells.append(take(69))  # final report print code


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7: Conclusions & Limitations
# ─────────────────────────────────────────────────────────────────────────────
new_cells.append(md("""\
<a id='conclusions'></a>

---
## 7. Conclusions & Limitations\
"""))

new_cells.append(rewrite(83, """\
### 7.1 Key Findings

**Mean predictability is limited — as expected.**
Daily log-returns are close to white noise, consistent with the weak-form efficient market
hypothesis. All models achieve MSE ≈ 0.00005 and directional accuracy in the 54–58% range.
A 4–8 pp edge above 50% is modest but economically meaningful when compounded over many trades.

**Two-step ARIMA-GARCH is well-specified.**
GARCH(1,1) successfully removes conditional heteroskedasticity: ARCH-LM on standardized
residuals fails to reject no-ARCH at 5%. Residual diagnostics (Ljung-Box, Q-Q) confirm
no egregious misspecification. The model is a sound statistical baseline.

**Sector ETF exogenous block adds directional edge.**
ARX-GARCH models match or exceed the pure ARIMA baseline on directional accuracy while also
producing calibrated probabilistic forecasts. Lagged sector ETF returns carry cross-sectional
momentum and rotation signals that univariate ARIMA cannot access.

**PCA + EN vs PLS: similar accuracy, different stability profile.**
Both compression methods achieve similar out-of-sample MSE and directional accuracy.
PCA+EN (unsupervised) is more robust to collinearity and noise. PLS (supervised) is
more sample-efficient but risks instability when the ETF-return relationship shifts over time.
Diagnostic loading tables in Section 5 identify which sectors drive each model's latent structure.

**FRED macro extension: controlled experiment.**
Adding 5 daily macro change variables expands the exogenous block from 55 → 80 features.
See Section 6.3 for the specific deltas in MSE, directional accuracy, and risk calibration.
The VIX change channel (ΔVIX) is the most likely improvement driver.

**Probabilistic outputs are the primary practical value.**
CI Coverage ≈ 96–97% and VaR Breach ≈ 3–5% confirm well-calibrated risk forecasts.
These enable dynamic position sizing, regime detection via σ spikes, and VaR reporting
consistent with Basel III IMA backtesting standards.

---

### 7.2 Limitations

| Limitation | Impact | Natural Fix |
|---|---|---|
| ARX ≠ ARIMAX | No MA terms; may leave serial correlation in mean residuals | `statsmodels ARIMA(endog, exog=PCA_factors)` for true ARIMAX |
| Single asset, 126-day test | Conclusions may not generalise; short window → noisy estimates | Block-bootstrap CI; multi-ticker study |
| No transaction costs | Direction accuracy overstates tradeable edge | Subtract 2× half-spread per round-trip |
| GARCH(1,1) symmetric | No leverage effect (negative returns → asymmetrically higher vol) | GJR-GARCH or EGARCH |
| Fixed XGBoost/LSTM params | No systematic hyperparameter optimisation | Time-series CV or Bayesian optimisation |
| FRED forward-fill | Weekend/holiday gaps filled with last change — mild approximation | FRED vintage calendars for precise release timing |

---

### 7.3 Extensions

1. **True ARIMAX-GARCH** — `statsmodels ARIMA(endog, exog=pca_factors, order=(p,0,q))` for
   joint MLE of AR, MA, and exog coefficients, then GARCH on ARIMAX residuals

2. **Cross-sectional DCC-GARCH** — apply the pipeline to a portfolio of sector ETFs and
   model return correlations with Dynamic Conditional Correlation GARCH

3. **Regime-switching GARCH** — Markov-switching GARCH for explicit high-vol / low-vol
   regimes; multimodal forecast distributions during regime transitions

4. **Options-implied features** — VIX term structure (VIX9D, VIX, VIX3M) and risk-neutral
   skewness from the options surface as additional exogenous variables

5. **Dynamic factor model** — Kalman filter to allow PCA/PLS loadings to evolve over time,
   relaxing the implicit stationarity assumption in the fixed walk-forward design

6. **Transformer sequence model** — temporal attention to replace LSTM for better handling
   of non-stationary financial time series and long-range dependencies\
"""))


# ══════════════════════════════════════════════════════════════════════════════
# WRITE NEW NOTEBOOK
# ══════════════════════════════════════════════════════════════════════════════
nb["cells"] = new_cells

with open(NB, "w") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"✓ Restructured notebook written.")
print(f"  Original cells : 84")
print(f"  Cells dropped  : 3  (orig[39], orig[46], orig[63])")
print(f"  New cells added: {len(new_cells) - (84 - 3)}")
print(f"  Total cells    : {len(new_cells)}")

# Print new cell structure for verification
print("\nNew notebook structure:")
for i, c in enumerate(new_cells):
    src = "".join(c["source"])
    kind = c["cell_type"][:2].upper()
    print(f"  [{i:3d}] {kind}  {src[:80].strip()!r}")
