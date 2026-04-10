"""
Injects ARIMAX-GARCH extension cells into forecasting_analysis.ipynb.
Run with the BT3102 Python:
  /Users/zimo/miniconda3/envs/bt3102/bin/python inject_cells.py
"""

import json

NB_PATH = "/Users/zimo/mmforecasting/notebooks/forecasting_analysis.ipynb"

# ── helpers ───────────────────────────────────────────────────────────────────

def code(src: str) -> dict:
    return {"cell_type": "code", "execution_count": None,
            "metadata": {}, "outputs": [], "source": src}

def md(src: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": src}

# ── load ──────────────────────────────────────────────────────────────────────

with open(NB_PATH) as f:
    nb = json.load(f)

cells = nb["cells"]
print(f"Loaded notebook: {len(cells)} cells")

# ═══════════════════════════════════════════════════════════════════════════════
# Cell content definitions
# ═══════════════════════════════════════════════════════════════════════════════

# ── G1: Additional imports (insert after cell 4 — the main imports cell) ──────

G1a = code("""\
# Additional imports for ARIMAX-GARCH models
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import ElasticNetCV, LinearRegression
from collections import Counter
print("✓ ARIMAX-GARCH imports OK")
""")

# ── G2: Exogenous data setup (insert after cell 7 — df = fetch_data(...)) ─────

G2_md = md("""\
---
# Part E: Exogenous Variables — Market + Sector ETFs

Sector ETFs (SPDR set) are used as lagged exogenous predictors in ARIMAX-GARCH.

**Self-regression rule:**
- If target = market index (SPY) → exclude SPY from exogenous set
- If target = sector ETF (e.g. XLK) → exclude that same ETF
- If target = individual stock → all tickers allowed
""")

G2b = code("""\
# ── Exogenous variable configuration ──────────────────────────────────────────
MARKET_TICKER_EX = "SPY"
SECTOR_ETFS_EX   = ["XLB","XLE","XLF","XLI","XLK","XLP","XLRE","XLU","XLV","XLY","XLC"]
ALL_EXOG_CANDIDATES = [MARKET_TICKER_EX] + SECTOR_ETFS_EX

N_AR_LAGS_EX = 5   # AR lags of target to include as candidates
N_EXOG_LAGS  = 5   # lag depth per exogenous series
PCA_K        = 3   # PCA components (Model 1)
N_COMP_PLS   = 3   # PLS components (Model 2)
N_AR_FIXED   = 3   # fixed AR lags used in PLS mean equation (lags 1..N_AR_FIXED)

# ── Step 2: Self-regression check ─────────────────────────────────────────────
def determine_exog_tickers(target):
    target_u = target.upper()
    included, excluded, reasons = [], [], {}
    for t in ALL_EXOG_CANDIDATES:
        if t.upper() == target_u:
            excluded.append(t)
            reasons[t] = "Self-regression exclusion: matches target ticker"
        else:
            included.append(t)
    return included, excluded, reasons

EXOG_INCLUDED, EXOG_EXCLUDED, EXOG_REASONS = determine_exog_tickers(TICKER)

# Classify target
if TICKER.upper() == MARKET_TICKER_EX:
    _ttype = "market index"
elif TICKER.upper() in SECTOR_ETFS_EX:
    _ttype = "sector ETF"
else:
    _ttype = "individual stock"

print("=" * 60)
print("EXOGENOUS VARIABLE CONFIGURATION")
print("=" * 60)
print(f"Target: {TICKER} ({_ttype})")
print(f"\\nIncluded exogenous variables ({len(EXOG_INCLUDED)}):")
for t in EXOG_INCLUDED:
    src = "market" if t == MARKET_TICKER_EX else "sector ETF"
    print(f"  + {t:<8} [{src}]")
if EXOG_EXCLUDED:
    print(f"\\nExcluded variables ({len(EXOG_EXCLUDED)}):")
    for t in EXOG_EXCLUDED:
        print(f"  - {t:<8} Reason: {EXOG_REASONS[t]}")
print("=" * 60)
""")

G2c = code("""\
# ── Fetch exogenous returns ────────────────────────────────────────────────────
import yfinance as yf

print(f"Fetching exog data for {len(EXOG_INCLUDED)} tickers...")
_exog_raw = yf.download(EXOG_INCLUDED, start=START_DATE, end=END_DATE,
                         progress=False, auto_adjust=True)
if isinstance(_exog_raw.columns, pd.MultiIndex):
    _exog_prices = _exog_raw["Close"]
else:
    _exog_prices = _exog_raw[["Close"]].copy()
    _exog_prices.columns = EXOG_INCLUDED

_exog_prices.columns = [str(c) for c in _exog_prices.columns]
_exog_prices = _exog_prices.dropna(how="all")
exog_returns = np.log(_exog_prices / _exog_prices.shift(1)).dropna(how="all")

print(f"✓ Exog returns: {exog_returns.shape[0]} obs × {exog_returns.shape[1]} tickers")
print(f"  Range: {exog_returns.index[0].date()} – {exog_returns.index[-1].date()}")
print(f"  Tickers: {list(exog_returns.columns)}")
""")

G2d = code("""\
# ── Step 3: Feature construction — lagged returns design matrix ───────────────
def build_design_matrix(target_series, exog_rets, n_ar, n_ex):
    \"\"\"
    Build joint aligned design matrix (y, X_ar, X_exog_lags).
    All features are strictly lagged — zero lookahead.
      y_lag1 .. y_lag{n_ar}         : AR lags of target
      {ticker}_lag1 .. lag{n_ex}    : lagged exog returns per ticker
    \"\"\"
    ar_df = pd.DataFrame(
        {f"y_lag{i}": target_series.shift(i) for i in range(1, n_ar + 1)},
        index=target_series.index
    )
    ex_parts = []
    for col in exog_rets.columns:
        for lag in range(1, n_ex + 1):
            s = exog_rets[col].shift(lag)
            s.name = f"{col}_lag{lag}"
            ex_parts.append(s)
    ex_df = pd.concat(ex_parts, axis=1)
    combined = pd.concat([target_series.rename("y"), ar_df, ex_df], axis=1).dropna()
    return combined["y"], combined[list(ar_df.columns)], combined[list(ex_df.columns)]

y_dm, X_ar_dm, X_exog_dm = build_design_matrix(
    df["Return"], exog_returns, N_AR_LAGS_EX, N_EXOG_LAGS
)

raw_exog_feat = len(EXOG_INCLUDED) * N_EXOG_LAGS
print("=" * 60)
print("DESIGN MATRIX SUMMARY")
print("=" * 60)
print(f"Aligned observations : {len(y_dm)}")
print(f"Effective start      : {y_dm.index[0].date()}")
print(f"Test window          : {y_dm.index[-TEST_SIZE].date()} – {y_dm.index[-1].date()}")
print(f"AR features          : {X_ar_dm.shape[1]}  (y_lag1..y_lag{N_AR_LAGS_EX})")
print(f"Exog features        : {X_exog_dm.shape[1]}  "
      f"({len(EXOG_INCLUDED)} tickers × {N_EXOG_LAGS} lags = {raw_exog_feat})")
print(f"PCA reduces exog to  : {PCA_K} components")
print(f"PLS uses             : {N_COMP_PLS} latent components")
print("=" * 60)
""")

# ── G3: ARIMAX-GARCH models (insert after LSTM results, currently cell 52) ────

G3_md = md("""\
---
# Part F: ARIMAX-GARCH Models

Two ARIMAX-GARCH variants replace classical ARIMA order selection with
data-driven dimensionality reduction on the exogenous sector ETF lag block.

### Model 1 — ARX-GARCH | PCA + Elastic Net
1. **PCA** (k components) reduces 55 exog lag features on the training window only
2. **Elastic Net** selects sparse subset of `[AR lags 1–5 of target]` + `[k PCA factors]`
   *(EN performs lag selection, not classical ARIMA order estimation)*
3. **GARCH(1,1)** fitted on mean-model residuals for variance

### Model 2 — ARX-GARCH | PLS
1. **PLS** (c supervised components) extracts latent factors from exog lag block
2. **OLS**: `y ~ [fixed AR lags 1..N_AR_FIXED, PLS scores]`
   *(PLS does not select AR lags; a fixed small candidate set is used)*
3. **GARCH(1,1)** fitted on OLS residuals for variance

### Forecast outputs (all steps)
`pred_mean` · `pred_variance` · `pred_std` · `ci_95_lower` · `ci_95_upper` · `var_95` · `var_99`

CI definition: μ ± 1.96σ
VaR definition: VaR₉₅ = −(μ + z₀.₀₅ · σ)  =  −μ + 1.645σ
""")

G3b = code("""\
# ── ARIMAX configuration & shared helpers ────────────────────────────────────
DIST_ARIMAX = BEST_DIST   # inherit distribution from baseline GARCH selection
print(f"GARCH distribution for ARIMAX models : {DIST_ARIMAX}")
print(f"PCA components k                     : {PCA_K}")
print(f"PLS components c                     : {N_COMP_PLS}")
print(f"Fixed AR lags (PLS mean equation)    : 1..{N_AR_FIXED}")

def _arimax_forecast_record(date, actual, mu, sigma):
    \"\"\"Build probabilistic forecast dict with CI and VaR.\"\"\"
    ci_lo = mu - 1.96 * sigma
    ci_hi = mu + 1.96 * sigma
    var95 = -(mu + stats.norm.ppf(0.05) * sigma)   # = −μ + 1.645σ
    var99 = -(mu + stats.norm.ppf(0.01) * sigma)   # = −μ + 2.326σ
    return {
        "Date": date, "actual": float(actual),
        "pred_mean": float(mu), "pred_variance": float(sigma**2),
        "pred_std": float(sigma),
        "ci_95_lower": float(ci_lo), "ci_95_upper": float(ci_hi),
        "var_95": float(var95), "var_99": float(var99),
    }

def _fit_arimax_garch_var(residuals, dist="t"):
    \"\"\"
    GARCH(1,1) on residuals with rescale=False.
    Returns one-step-ahead conditional variance in ORIGINAL scale.
    (rescale=False avoids the arch auto-scaling issue where forecast().variance
     is returned in rescaled units, not original units.)
    \"\"\"
    am = arch_model(residuals, mean="Zero", vol="Garch", p=1, q=1,
                    dist=dist, rescale=False)
    gfit = am.fit(disp="off", show_warning=False)
    fcast = gfit.forecast(horizon=1, reindex=False)
    return float(fcast.variance.iloc[-1, 0])
""")

G3c = code("""\
# ═══════════════════════════════════════════════════════════════════════════════
# Model 1: ARIMAX-GARCH (PCA + Elastic Net) — walk-forward backtest
# ═══════════════════════════════════════════════════════════════════════════════
# At each step:
#   1. PCA (k components) on exog lag matrix — fit on training data only
#   2. X_candidate = [AR lags 1-5, PCA factors]  →  StandardScaler
#   3. ElasticNetCV: sparse selection over candidate predictors (lag selection)
#   4. mu_hat = EN.predict(test row)
#   5. residuals = y_train - EN.predict(X_train)
#   6. GARCH(1,1) on residuals  →  one-step-ahead variance
#   7. sigma_hat = sqrt(var_hat)

train_end_dm = len(y_dm) - TEST_SIZE
pca_en_records  = []
pca_en_fail     = 0
pca_en_ar_hist  = []   # track selected AR lags per step

_last_mu_pe  = 0.0
_last_sig_pe = float(np.std(y_dm.iloc[:train_end_dm].values))

for i in tqdm(range(TEST_SIZE), desc="ARIMAX-GARCH (PCA+EN)"):
    actual = float(y_dm.iloc[train_end_dm + i])
    date   = y_dm.index[train_end_dm + i]

    y_tr  = y_dm.iloc[:train_end_dm + i].values.astype(float)
    ar_tr = X_ar_dm.iloc[:train_end_dm + i].values.astype(float)
    ex_tr = X_exog_dm.iloc[:train_end_dm + i].values.astype(float)
    ar_te = X_ar_dm.iloc[[train_end_dm + i]].values.astype(float)
    ex_te = X_exog_dm.iloc[[train_end_dm + i]].values.astype(float)

    try:
        # 1. PCA on exog (training only)
        k = min(PCA_K, ex_tr.shape[1], ex_tr.shape[0] - 1)
        pca = PCA(n_components=k, random_state=RANDOM_SEED)
        fac_tr = pca.fit_transform(ex_tr)
        fac_te = pca.transform(ex_te)

        # 2. Candidate matrix: AR lags + PCA factors
        X_cand_tr = np.hstack([ar_tr, fac_tr])
        X_cand_te = np.hstack([ar_te, fac_te])
        scaler_pe = StandardScaler()
        X_sc_tr = scaler_pe.fit_transform(X_cand_tr)
        X_sc_te = scaler_pe.transform(X_cand_te)

        # 3. Elastic Net CV — sparse lag selection
        #    Selects among AR lags AND PCA factors; NOT classical ARIMA order estimation
        en = ElasticNetCV(l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 1.0],
                          cv=3, max_iter=10_000,
                          random_state=RANDOM_SEED, n_jobs=1)
        en.fit(X_sc_tr, y_tr)

        # 4. Mean forecast
        mu_hat = float(en.predict(X_sc_te)[0])

        # Track selected AR lags (first N_AR_LAGS_EX coefs map to AR lag 1..5)
        ar_coef = en.coef_[:N_AR_LAGS_EX]
        sel_ar  = [j + 1 for j, c in enumerate(ar_coef) if abs(c) > 1e-10]
        pca_en_ar_hist.append(sel_ar)

        # 5–6. Residuals → GARCH
        resid   = y_tr - en.predict(X_sc_tr)
        var_hat = _fit_arimax_garch_var(resid, dist=DIST_ARIMAX)
        sigma_hat = float(np.sqrt(max(var_hat, 1e-12)))

        _last_mu_pe, _last_sig_pe = mu_hat, sigma_hat

    except Exception:
        pca_en_fail += 1
        mu_hat, sigma_hat, sel_ar = _last_mu_pe, _last_sig_pe, []
        pca_en_ar_hist.append(sel_ar)

    rec = _arimax_forecast_record(date, actual, mu_hat, sigma_hat)
    rec["selected_ar_lags"] = sel_ar
    pca_en_records.append(rec)

pca_en_results = pd.DataFrame(pca_en_records).set_index("Date")

# Derive canonical model name: AR lags appearing in >50% of backtest steps
_all_ar = [l for lags in pca_en_ar_hist for l in lags]
_cnt    = Counter(_all_ar)
_freq   = sorted(l for l, c in _cnt.items() if c > 0.5 * TEST_SIZE)
_lag_str_pe = ",".join(str(l) for l in _freq) if _freq else "none"
PCA_EN_NAME = f"ARX[{_lag_str_pe}]-GARCH(1,1) | PCA(k={PCA_K})+EN"

print(f"\\nBacktest done. Failures carried forward: {pca_en_fail}/{TEST_SIZE}")
print(f"Model: {PCA_EN_NAME}")
""")

G3d = code("""\
# ── PCA+EN results ────────────────────────────────────────────────────────────
_pe_mse = float(np.mean((pca_en_results["actual"] - pca_en_results["pred_mean"])**2))
_pe_mae = float(np.mean(np.abs(pca_en_results["actual"] - pca_en_results["pred_mean"])))
_pe_dir = float(np.mean(np.sign(pca_en_results["actual"]) == np.sign(pca_en_results["pred_mean"])))

print(f"=== {PCA_EN_NAME} ===")
print(f"MSE:           {_pe_mse:.8f}")
print(f"MAE:           {_pe_mae:.8f}")
print(f"Direction Acc: {_pe_dir:.2%}")
print()
display(pca_en_results[["actual","pred_mean","pred_std",
                          "ci_95_lower","ci_95_upper","var_95","var_99"]].head(8).round(6))
""")

G3e = code("""\
# ═══════════════════════════════════════════════════════════════════════════════
# Model 2: ARIMAX-GARCH (PLS) — walk-forward backtest
# ═══════════════════════════════════════════════════════════════════════════════
# At each step:
#   1. PLS (c components) on exog lag matrix supervised by y — training only
#   2. PLS scores (X-side latent vars) combined with fixed AR lags
#   3. OLS: y ~ [AR lags 1..N_AR_FIXED, PLS scores]
#   4. mu_hat = OLS.predict(test row)
#   5. residuals = y_train - OLS.predict(X_train)
#   6. GARCH(1,1) on residuals  →  one-step-ahead variance

pls_records = []
pls_fail    = 0

_last_mu_pls  = 0.0
_last_sig_pls = float(np.std(y_dm.iloc[:train_end_dm].values))

for i in tqdm(range(TEST_SIZE), desc="ARIMAX-GARCH (PLS)"):
    actual = float(y_dm.iloc[train_end_dm + i])
    date   = y_dm.index[train_end_dm + i]

    y_tr  = y_dm.iloc[:train_end_dm + i].values.astype(float)
    ar_tr = X_ar_dm.iloc[:train_end_dm + i, :N_AR_FIXED].values.astype(float)
    ex_tr = X_exog_dm.iloc[:train_end_dm + i].values.astype(float)
    ar_te = X_ar_dm.iloc[[train_end_dm + i], :N_AR_FIXED].values.astype(float)
    ex_te = X_exog_dm.iloc[[train_end_dm + i]].values.astype(float)

    try:
        # 1. PLS on exog supervised by y (training only)
        nc  = min(N_COMP_PLS, ex_tr.shape[1], ex_tr.shape[0] - 1)
        pls = PLSRegression(n_components=nc, scale=True)
        pls.fit(ex_tr, y_tr)
        pls_sc_tr = pls.transform(ex_tr)    # (n_train, nc) X-side scores
        pls_sc_te = pls.transform(ex_te)    # (1, nc)

        # 2–3. OLS: y ~ [fixed AR lags, PLS scores]
        X_ols_tr = np.hstack([ar_tr, pls_sc_tr])
        X_ols_te = np.hstack([ar_te, pls_sc_te])
        ols = LinearRegression(fit_intercept=True)
        ols.fit(X_ols_tr, y_tr)

        # 4. Mean forecast
        mu_hat = float(ols.predict(X_ols_te)[0])

        # 5–6. Residuals → GARCH
        resid   = y_tr - ols.predict(X_ols_tr)
        var_hat = _fit_arimax_garch_var(resid, dist=DIST_ARIMAX)
        sigma_hat = float(np.sqrt(max(var_hat, 1e-12)))

        _last_mu_pls, _last_sig_pls = mu_hat, sigma_hat

    except Exception:
        pls_fail += 1
        mu_hat, sigma_hat = _last_mu_pls, _last_sig_pls

    pls_records.append(_arimax_forecast_record(date, actual, mu_hat, sigma_hat))

pls_results = pd.DataFrame(pls_records).set_index("Date")

_lag_str_pls = ",".join(str(i) for i in range(1, N_AR_FIXED + 1))
PLS_NAME = f"ARX[{_lag_str_pls}]-GARCH(1,1) | PLS(c={N_COMP_PLS})"

print(f"\\nBacktest done. Failures carried forward: {pls_fail}/{TEST_SIZE}")
print(f"Model: {PLS_NAME}")
""")

G3f = code("""\
# ── PLS results ───────────────────────────────────────────────────────────────
_pls_mse = float(np.mean((pls_results["actual"] - pls_results["pred_mean"])**2))
_pls_mae = float(np.mean(np.abs(pls_results["actual"] - pls_results["pred_mean"])))
_pls_dir = float(np.mean(np.sign(pls_results["actual"]) == np.sign(pls_results["pred_mean"])))

print(f"=== {PLS_NAME} ===")
print(f"MSE:           {_pls_mse:.8f}")
print(f"MAE:           {_pls_mae:.8f}")
print(f"Direction Acc: {_pls_dir:.2%}")
print()
display(pls_results[["actual","pred_mean","pred_std",
                       "ci_95_lower","ci_95_upper","var_95","var_99"]].head(8).round(6))
""")

# ── G4: Extended model comparison (append after existing comparison) ───────────

G4_md = md("""\
---
# Part G: Extended Model Comparison — All Models

Metrics for the two new ARIMAX-GARCH models alongside the three baseline models.

**New metrics (ARIMAX models only, require probabilistic forecasts):**
- **CI Coverage**: % of actuals inside predicted 95% CI — target ≈ 95%
- **VaR Breach**: % of actuals breaching the 95% VaR — target ≈ 5%

**Breach condition**: actual return < −VaR₉₅ = μ − 1.645σ
""")

G4b = code("""\
# ── Full metrics computation for all models ───────────────────────────────────

def compute_full_metrics(forecast_df):
    \"\"\"Metrics for probabilistic forecast models (ARIMAX variants).\"\"\"
    y   = forecast_df["actual"].values.astype(float)
    mu  = forecast_df["pred_mean"].values.astype(float)
    clo = forecast_df["ci_95_lower"].values.astype(float)
    chi = forecast_df["ci_95_upper"].values.astype(float)
    v95 = forecast_df["var_95"].values.astype(float)
    return {
        "MSE":               float(np.mean((y - mu)**2)),
        "MAE":               float(np.mean(np.abs(y - mu))),
        "Dir Acc":           float(np.mean(np.sign(y) == np.sign(mu))),
        "CI Coverage (95%)": float(np.mean((y >= clo) & (y <= chi))),
        "VaR Breach (95%)":  float(np.mean(y < -v95)),
    }

def compute_point_metrics(actual, predicted):
    \"\"\"Metrics for point-forecast-only models (no CI/VaR).\"\"\"
    y, mu = np.asarray(actual, float), np.asarray(predicted, float)
    return {
        "MSE":               float(np.mean((y - mu)**2)),
        "MAE":               float(np.mean(np.abs(y - mu))),
        "Dir Acc":           float(np.mean(np.sign(y) == np.sign(mu))),
        "CI Coverage (95%)": float("nan"),
        "VaR Breach (95%)":  float("nan"),
    }

# Collect all model metrics
_baseline_name = f"ARIMA({BEST_P},{BEST_D},{BEST_Q})-GARCH(1,1)-{BEST_DIST}"
all_metrics = {
    _baseline_name: compute_point_metrics(
        arima_results["Actual"], arima_results["ARIMA_GARCH_Pred"]
    ),
    "XGBoost": compute_point_metrics(
        xgb_results["Actual"], xgb_results["XGB_Pred"]
    ),
    "LSTM": compute_point_metrics(
        lstm_results["Actual"], lstm_results["LSTM_Pred"]
    ),
    PCA_EN_NAME: compute_full_metrics(pca_en_results),
    PLS_NAME:    compute_full_metrics(pls_results),
}

# ── Print structured report ───────────────────────────────────────────────────
W = 75
print()
print("=" * W)
print(f"MODEL COMPARISON REPORT: {TICKER}")
print("=" * W)
print(f"\\nBacktest: {TEST_SIZE} days, walk-forward expanding window\\n")

_hdr = (f"{'Model':<42} {'MSE':>9} {'Dir Acc':>9} "
        f"{'CI Cov':>9} {'VaR Breach':>11}")
print(_hdr)
print("-" * W)

def _fmt(v, pct=False):
    if v != v:   # nan check
        return "    N/A"
    return f"{v:>8.1%}" if pct else f"{v:>9.5f}"

for name, m in all_metrics.items():
    print(f"{name:<42} "
          f"{_fmt(m['MSE'])} "
          f"{_fmt(m['Dir Acc'], pct=True)} "
          f"{_fmt(m['CI Coverage (95%)'], pct=True)} "
          f"{_fmt(m['VaR Breach (95%)'], pct=True)}")

print("-" * W)
print()
print("Metric Interpretation:")
print("  MSE        : lower = better point-forecast accuracy")
print("  Dir Acc    : % correct sign prediction  (50% = random, 52-55% = meaningful edge)")
print("  CI Cov     : % actuals inside 95% CI    (target ≈ 95%)  [ARIMAX models only]")
print("  VaR Breach : % actuals breaching VaR₉₅  (target ≈  5%)  [ARIMAX models only]")

# Winners
def _best(key, hi=False):
    c = {k: v[key] for k, v in all_metrics.items() if v[key] == v[key]}
    return (max if hi else min)(c, key=c.__getitem__) if c else "N/A"

def _risk_best():
    c = {k: abs(v.get("CI Coverage (95%)", float("nan")) - 0.95)
             + abs(v.get("VaR Breach (95%)", float("nan")) - 0.05)
         for k, v in all_metrics.items()
         if v.get("CI Coverage (95%)", float("nan")) == v.get("CI Coverage (95%)", float("nan"))}
    return min(c, key=c.__getitem__) if c else "N/A"

print()
print("Winners:")
print(f"  Best Forecast Accuracy (MSE) : {_best('MSE')}")
print(f"  Best Direction Prediction    : {_best('Dir Acc', hi=True)}")
print(f"  Best Risk Calibration        : {_risk_best()}")
print("=" * W)

# Return as DataFrame
ext_metrics_df = pd.DataFrame(all_metrics).T
display(ext_metrics_df.round(5))
""")

G4c = code("""\
# ── Forecast visualization: ARIMAX models with CI bands ──────────────────────
fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

for ax, results_df, name, color in zip(
    axes,
    [pca_en_results, pls_results],
    [PCA_EN_NAME, PLS_NAME],
    ["steelblue", "darkorange"]
):
    ax.plot(results_df.index, results_df["actual"],
            "k-", lw=1.5, alpha=0.85, label="Actual return")
    ax.plot(results_df.index, results_df["pred_mean"],
            color=color, ls="--", lw=1.2, alpha=0.9, label="Predicted mean")
    ax.fill_between(
        results_df.index,
        results_df["ci_95_lower"], results_df["ci_95_upper"],
        alpha=0.20, color=color, label="95% CI"
    )
    # Mark VaR breaches (actual < -VaR_95)
    breaches = results_df[results_df["actual"] < -results_df["var_95"]]
    ax.scatter(breaches.index, breaches["actual"],
               color="red", zorder=5, s=40, label=f"VaR breach ({len(breaches)})")
    ax.axhline(0, color="grey", ls=":", alpha=0.4)
    ax.set_title(name, fontsize=11)
    ax.set_ylabel("Log Return")
    ax.legend(loc="upper right", fontsize=9)

axes[-1].set_xlabel("Date")
plt.suptitle(f"{TICKER} — ARIMAX-GARCH Forecasts vs Actual  ({TEST_SIZE}-day backtest)",
             fontsize=12, y=1.01)
plt.tight_layout()
plt.show()
""")

G4d = code("""\
# ── Predicted volatility comparison ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(pca_en_results.index, pca_en_results["pred_std"] * np.sqrt(252),
        label=f"{PCA_EN_NAME} — Ann. Vol", color="steelblue", lw=1.2)
ax.plot(pls_results.index, pls_results["pred_std"] * np.sqrt(252),
        label=f"{PLS_NAME} — Ann. Vol", color="darkorange", ls="--", lw=1.2)
ax.axhline(pca_en_results["pred_std"].mean() * np.sqrt(252),
           color="steelblue", ls=":", alpha=0.5)
ax.axhline(pls_results["pred_std"].mean() * np.sqrt(252),
           color="darkorange", ls=":", alpha=0.5)
ax.set_title("Predicted Annualised Volatility — ARIMAX-GARCH Models")
ax.set_ylabel("Annualised Volatility")
ax.legend()
plt.tight_layout()
plt.show()

print(f"\\nAverage predicted daily vol (PCA+EN): {pca_en_results['pred_std'].mean():.4f}")
print(f"Average predicted daily vol (PLS)    : {pls_results['pred_std'].mean():.4f}")
print(f"Realised daily vol (test period)     : {pca_en_results['actual'].std():.4f}")
""")

# ═══════════════════════════════════════════════════════════════════════════════
# Inject all new cell groups into the notebook
# Insert in REVERSE positional order so earlier indices remain valid
# ═══════════════════════════════════════════════════════════════════════════════

# Original cell indices (0-based) BEFORE any insertions:
# 4  — main imports (insert G1 after this)
# 7  — df = fetch_data(...) (insert G2 after this)
# 52 — lstm_results DataFrame + metrics (insert G3 after this)
# 59 — final report print (insert G4 after this)

insertions = [
    (59, [G4_md, G4b, G4c, G4d]),   # Extended comparison
    (52, [G3_md, G3b, G3c, G3d, G3e, G3f]),  # ARIMAX models
    (7,  [G2_md, G2b, G2c, G2d]),   # Exog data setup
    (4,  [G1a]),                     # Additional imports
]

for idx, new_cells in insertions:
    for offset, cell in enumerate(new_cells):
        cells.insert(idx + 1 + offset, cell)
    print(f"  Inserted {len(new_cells)} cell(s) after original index {idx}")

print(f"\\nTotal cells after injection: {len(cells)}")

# ── write back ────────────────────────────────────────────────────────────────
with open(NB_PATH, "w") as f:
    json.dump(nb, f, indent=1)

print(f"✓ Notebook saved: {NB_PATH}")
