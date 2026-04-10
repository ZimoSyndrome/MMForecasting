"""
Adds methodology markdown, changes PCA_K=5, and enhances diagnostic output
for the ARIMAX-GARCH cells in forecasting_analysis.ipynb.

Run with:
  /Users/zimo/miniconda3/envs/bt3102/bin/python update_arimax_cells.py
"""
import json

NB_PATH = "/Users/zimo/mmforecasting/notebooks/forecasting_analysis.ipynb"

with open(NB_PATH) as f:
    nb = json.load(f)
cells = nb["cells"]
print(f"Loaded: {len(cells)} cells")

def code(src): return {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":src}
def md(src):   return {"cell_type":"markdown","metadata":{},"source":src}

def find_idx(marker, cell_type="code"):
    for i, c in enumerate(cells):
        if c["cell_type"] == cell_type and marker in "".join(c["source"]):
            return i
    return None

# ═══════════════════════════════════════════════════════════════════════════════
# 1.  PCA_K = 3 → 5
# ═══════════════════════════════════════════════════════════════════════════════
idx = find_idx("PCA_K        = 3")
assert idx is not None, "Could not find PCA_K config cell"
cells[idx]["source"] = "".join(cells[idx]["source"]).replace(
    "PCA_K        = 3", "PCA_K        = 5"
)
print(f"[{idx}] PCA_K updated to 5")

# ═══════════════════════════════════════════════════════════════════════════════
# 2.  Replace G3_md (ARIMAX section overview)
# ═══════════════════════════════════════════════════════════════════════════════
idx_g3md = find_idx("Part F: ARIMAX-GARCH Models", "markdown")
assert idx_g3md is not None
cells[idx_g3md]["source"] = """\
---
# Part F: ARIMAX-GARCH Models

Classical ARIMA-GARCH uses a grid search over AR/MA orders (AIC/BIC) and ignores the rich information in related markets. The two models in this section **extend** that framework by incorporating sector ETF returns as exogenous predictors — while keeping the same walk-forward, no-lookahead discipline.

Both models share the same two-step structure:
1. **Mean equation** — a linear combination of lagged target returns + compressed sector factors
2. **Variance equation** — GARCH(1,1) on the mean-model residuals

What differs is *how* the 55 exogenous lag features are compressed before entering the mean equation:

| | PCA + Elastic Net | PLS |
|---|---|---|
| Compression | Unsupervised (maximise sector variance) | Supervised (maximise covariance with y) |
| Selection | Elastic Net prunes AR lags + PCA factors | Fixed AR set; all PLS scores included |
| AR lags | Data-driven (EN selects from lags 1–5) | Fixed (lags 1–{N_AR_FIXED}) |
| Components | k = 5 PCA components | c = 3 PLS components |

**Forecast outputs (all steps):**
`pred_mean` · `pred_variance` · `pred_std` · `ci_95_lower` · `ci_95_upper` · `var_95` · `var_99`
"""
print(f"[{idx_g3md}] G3_md updated")

# ═══════════════════════════════════════════════════════════════════════════════
# 3.  New markdown cells to INSERT (content defined first, inserted later)
# ═══════════════════════════════════════════════════════════════════════════════

EXOG_MD = md("""\
---
### How Exogenous Variables Are Constructed

**Why sector ETFs?**
Equity markets are segmented by industry. Technology, Financials, Energy, and
Utilities do not all move in lockstep — they respond differently to rate changes,
commodity prices, and earnings cycles. By including lagged returns from the 11
SPDR sector ETFs we give the model a window into **industry momentum and
rotation effects** that a pure autoregressive model on SPY alone cannot see.

**Tickers included (self-regression rule applied):**
When the target is SPY (the broad market), SPY itself is excluded from the
exogenous set to prevent self-regression. All 11 sector ETFs are included.
When the target is a sector ETF, that same ticker is excluded instead.

**The lagging rule — strict no-lookahead:**
For a forecast on day *t + 1* we only use information available at the *close*
of day *t* or earlier. In practice this means sector returns enter the model
at lags 1 through 5:

    Exog feature at time t  →  ETF_return_{t-1}, ETF_return_{t-2}, ..., ETF_return_{t-5}

Same-day sector returns (lag 0) are excluded: they would require knowing today's
closing prices, which are not yet available when we submit the forecast.

**Feature dimensions:**

| Block | Description | Count |
|---|---|---|
| AR lags | y_lag1 … y_lag5 (target return lags) | 5 |
| Exog lags | 11 ETFs × 5 lags | 55 |
| **Total candidates** | | **60** |

**Data alignment note:**
XLC (Communication Services) was launched in June 2018. Because the full 60-column
design matrix requires all 11 sector ETFs to be non-missing, the effective training
start shifts to mid-2018 after `dropna()`. Approximately 6.5 years of prior SPY
history is sacrificed to keep the panel balanced. The test window (last 126 trading
days) is unaffected — it is well within the XLC history.
""")

PCA_EN_MD = md("""\
---
### Model 1 — ARX-GARCH | PCA + Elastic Net: Methodology

**The dimensionality challenge:**
With 55 exogenous lag features and roughly 1 800 training observations, a naive
OLS regression would overfit badly (high variance, poor out-of-sample performance).
We solve this with a two-stage pipeline:

---
#### Stage 1: Principal Component Analysis (unsupervised compression)

PCA finds orthogonal linear combinations of the 55 sector-lag columns that
explain the most variance in the *exogenous block itself* — without looking at
the target return y.

**Interpretation of components:**
- **PC1** typically captures **broad market co-movement** — a day where all
  sectors rise or fall together. This is the dominant mode of equity return variation.
- **PC2, PC3, ...** capture **rotation effects** — e.g. Energy outperforming
  Utilities, or Financials outperforming Defensives. These reflect structural
  shifts in risk appetite.

**Practical detail:**
PCA is fitted on the *training window only* at every walk-forward step. The
same eigenvectors are then applied to the test-row observation. This prevents
any information from future dates leaking into the component structure.

k = **5 components** are retained. The table below (printed after the backtest)
shows the average variance explained across all 126 walk-forward steps.

---
#### Stage 2: Elastic Net (sparse lag selection)

Elastic Net regression is then applied to a *combined* candidate matrix:

    X_candidate = [ y_lag1, y_lag2, y_lag3, y_lag4, y_lag5,
                    PC1,    PC2,    PC3,    PC4,    PC5    ]   (10 features total)

Elastic Net combines **Ridge** (L2) and **LASSO** (L1) penalties. The LASSO
component drives irrelevant coefficients to *exactly zero*, effectively selecting
which AR lags and PCA factors genuinely help out-of-sample.

> **Important:** Elastic Net here performs **data-driven lag selection** over the
> candidate set — playing the same role that AIC/BIC grid search plays for classical
> ARIMA, but without assuming a fixed model order and without enumerating all
> combinations. The result is a sparse ARX model whose structure adapts to the
> training data at each walk-forward step.

The regularisation strength α is chosen by **3-fold cross-validation** inside the
expanding training window. The L1 ratio is tuned over {0.1, 0.5, 0.7, 0.9, 0.95, 1.0}.

---
#### Volatility: GARCH(1,1)

GARCH(1,1) is fitted on the *Elastic Net residuals* (y_train − EN_prediction)
to capture the volatility clustering that the linear mean model leaves unexplained.
""")

PLS_MD = md("""\
---
### Model 2 — ARX-GARCH | PLS: Methodology

**The key difference from PCA+EN:**
PCA compressed the 55-column sector block by maximising *variance explained in X* —
a purely unsupervised criterion. Partial Least Squares (PLS) instead finds latent
components that maximise the **covariance between X scores and the target return y**.
The compression is *supervised*: every latent direction is explicitly chosen to be
as predictive of tomorrow's SPY return as possible.

---
#### How PLS Works Here

Given the 55-column exogenous lag matrix X_train and the target vector y_train, PLS
decomposes X into c latent *score vectors* T = X W* such that each column of T has
maximum covariance with y_train. Intuitively, each PLS component is a **portfolio of
sector-lag exposures** that historically co-moved most strongly with the next-day
broad market return.

c = **3 PLS components** are used. The diagnostic table below (printed after the
backtest) shows which sector ETFs load most heavily onto each component and reveals
the economic interpretation (e.g. momentum component, defensive rotation component).

PLS is fitted on *training data only* at every walk-forward step and the resulting
projections are applied to the test observation without refitting — maintaining strict
no-lookahead discipline.

---
#### Mean Equation Structure

Unlike PCA+EN, PLS does not perform AR lag selection. A **fixed small set of AR lags**
(lags 1 to {N_AR_FIXED}) is always included alongside the PLS scores in an OLS regression:

    y_t = β₀ + β₁·y_{t-1} + ... + β_p·y_{t-p}  +  γ₁·T₁ + γ₂·T₂ + γ₃·T₃  +  ε_t

where T₁, T₂, T₃ are the PLS score vectors. This separates the **lag-selection
concern** (handled by the fixed AR set) from the **dimension-reduction concern**
(handled by PLS).

---
#### Volatility: GARCH(1,1)

Same structure as Model 1 — GARCH(1,1) on the OLS residuals.

---
#### Why Compare PCA+EN vs PLS?

| Dimension | PCA + EN | PLS |
|---|---|---|
| Compression criterion | Max variance in X (unsupervised) | Max cov(X, y) (supervised) |
| AR lag selection | Data-driven (EN) | Fixed candidate set |
| Risk of overfitting | Regularised by EN | Low (only 3 components) |
| Interpretability | EN gives sparse coefficients | PLS loadings map to sectors |

Whether supervised compression (PLS) translates into better return forecasts than
unsupervised compression followed by sparse selection (PCA+EN) is an empirical
question — answered by the backtest comparison in Part G.
""")

# ═══════════════════════════════════════════════════════════════════════════════
# 4.  Updated PCA+EN backtest cell — adds diagnostic collection
# ═══════════════════════════════════════════════════════════════════════════════
PCA_EN_BACKTEST = code("""\
# ═══════════════════════════════════════════════════════════════════════════════
# Model 1: ARIMAX-GARCH (PCA + Elastic Net) — walk-forward backtest
# ═══════════════════════════════════════════════════════════════════════════════
# At each step:
#   1. PCA (k=5 components) on exog lag matrix — fit on training data only
#   2. X_candidate = [AR lags 1-5, PCA factors]  →  StandardScaler
#   3. ElasticNetCV: sparse selection over candidate predictors (lag selection)
#      Note: EN performs lag selection, NOT classical ARIMA order estimation
#   4. mu_hat = EN.predict(test row)
#   5. residuals = y_train - EN.predict(X_train)
#   6. GARCH(1,1) on residuals  →  one-step-ahead variance (rescale=False)
#   7. sigma_hat = sqrt(var_hat)
# Diagnostic data collected for post-hoc analysis of PCA structure & EN selection.

train_end_dm = len(y_dm) - TEST_SIZE
pca_en_records  = []
pca_en_fail     = 0
pca_en_ar_hist  = []    # selected AR lag indices per step

# Diagnostic collectors
_pca_var_ratios = []    # explained variance ratio per component (step, k)
_pca_comps_all  = []    # PCA components / loadings (step, k, n_exog)
_en_coefs_all   = []    # EN coefficients on scaled X_cand (step, n_ar+k)

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
        # 1. PCA on exog block (training only)
        k = min(PCA_K, ex_tr.shape[1], ex_tr.shape[0] - 1)
        pca = PCA(n_components=k, random_state=RANDOM_SEED)
        fac_tr = pca.fit_transform(ex_tr)
        fac_te = pca.transform(ex_te)

        _pca_var_ratios.append(pca.explained_variance_ratio_.copy())
        _pca_comps_all.append(pca.components_.copy())   # (k, n_exog)

        # 2. Candidate matrix: AR lags + PCA factors
        X_cand_tr = np.hstack([ar_tr, fac_tr])
        X_cand_te = np.hstack([ar_te, fac_te])
        scaler_pe = StandardScaler()
        X_sc_tr = scaler_pe.fit_transform(X_cand_tr)
        X_sc_te = scaler_pe.transform(X_cand_te)

        # 3. Elastic Net CV — sparse lag selection
        en = ElasticNetCV(l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 1.0],
                          cv=3, max_iter=10_000,
                          random_state=RANDOM_SEED, n_jobs=1)
        en.fit(X_sc_tr, y_tr)
        _en_coefs_all.append(en.coef_.copy())           # (n_ar + k,)

        # 4. Mean forecast
        mu_hat = float(en.predict(X_sc_te)[0])

        # Selected AR lags (first N_AR_LAGS_EX coefs in X_cand)
        ar_coef = en.coef_[:N_AR_LAGS_EX]
        sel_ar  = [j + 1 for j, c in enumerate(ar_coef) if abs(c) > 1e-10]
        pca_en_ar_hist.append(sel_ar)

        # 5–6. Residuals → GARCH (rescale=False keeps variance in original units)
        resid   = y_tr - en.predict(X_sc_tr)
        var_hat = _fit_arimax_garch_var(resid, dist=DIST_ARIMAX)
        sigma_hat = float(np.sqrt(max(var_hat, 1e-12)))

        _last_mu_pe, _last_sig_pe = mu_hat, sigma_hat

    except Exception:
        pca_en_fail += 1
        mu_hat, sigma_hat, sel_ar = _last_mu_pe, _last_sig_pe, []
        pca_en_ar_hist.append(sel_ar)
        # Append None placeholders so list lengths stay consistent
        if len(_pca_var_ratios) < i + 1:
            _pca_var_ratios.append(None)
            _pca_comps_all.append(None)
            _en_coefs_all.append(None)

    rec = _arimax_forecast_record(date, actual, mu_hat, sigma_hat)
    rec["selected_ar_lags"] = sel_ar
    pca_en_records.append(rec)

pca_en_results = pd.DataFrame(pca_en_records).set_index("Date")

# Derive canonical model name from AR lag selection history
_all_ar = [l for lags in pca_en_ar_hist for l in lags]
_cnt    = Counter(_all_ar)
_freq   = sorted(l for l, c in _cnt.items() if c > 0.5 * TEST_SIZE)
_lag_str_pe = ",".join(str(l) for l in _freq) if _freq else "none"
PCA_EN_NAME = f"ARX[{_lag_str_pe}]-GARCH(1,1) | PCA(k={PCA_K})+EN"

print(f"\\nBacktest done. Failures carried forward: {pca_en_fail}/{TEST_SIZE}")
print(f"Model: {PCA_EN_NAME}")
""")

# ═══════════════════════════════════════════════════════════════════════════════
# 5.  Updated PCA+EN results cell — adds diagnostic printout
# ═══════════════════════════════════════════════════════════════════════════════
PCA_EN_RESULTS = code("""\
# ── PCA+EN: point-forecast metrics ───────────────────────────────────────────
_pe_mse = float(np.mean((pca_en_results["actual"] - pca_en_results["pred_mean"])**2))
_pe_mae = float(np.mean(np.abs(pca_en_results["actual"] - pca_en_results["pred_mean"])))
_pe_dir = float(np.mean(np.sign(pca_en_results["actual"]) == np.sign(pca_en_results["pred_mean"])))

print(f"=== {PCA_EN_NAME} ===")
print(f"MSE:           {_pe_mse:.8f}")
print(f"MAE:           {_pe_mae:.8f}")
print(f"Direction Acc: {_pe_dir:.2%}")
print()
display(pca_en_results[["actual","pred_mean","pred_std",
                          "ci_95_lower","ci_95_upper","var_95","var_99"]].head(5).round(6))

# ── PCA+EN: diagnostic report ─────────────────────────────────────────────────
_valid_var   = [v for v in _pca_var_ratios if v is not None]
_valid_comps = [c for c in _pca_comps_all  if c is not None]
_valid_coefs = [c for c in _en_coefs_all   if c is not None]
_exog_names  = list(X_exog_dm.columns)       # ['XLB_lag1', 'XLB_lag2', ..., 'XLC_lag5']

print()
print("=" * 65)
print("PCA + ELASTIC NET — DIAGNOSTIC REPORT")
print("=" * 65)

# -- PCA variance explained ---------------------------------------------------
avg_var = np.mean(_valid_var, axis=0)
print(f"\\nExogenous block: {X_exog_dm.shape[1]} features  "
      f"({len(EXOG_INCLUDED)} sector ETFs × {N_EXOG_LAGS} lags)")
print(f"PCA retains k = {PCA_K} components.\\n")
print("Average variance explained (across all walk-forward steps):")
print(f"  {'Component':<14} {'Var Explained':>14} {'Cumulative':>12}")
print(f"  {'-'*42}")
cumvar = 0.0
for j, v in enumerate(avg_var):
    cumvar += v
    bar = "█" * int(v * 40)
    print(f"  PC{j+1:<12}   {v:>10.1%}     {cumvar:>10.1%}   {bar}")

# -- PCA loadings: top sector-lag features per component ----------------------
if _valid_comps:
    avg_comps = np.mean(_valid_comps, axis=0)   # (k, n_exog)
    print()
    print("Top sector-lag exposures per PCA component (avg |loading|):")
    print("  (Reveals which ETF lags define each component's economic meaning)")

    # Short ticker-only labels for readability
    def _short(name):
        parts = name.split("_")
        return f"{parts[0]} lag{parts[-1]}"

    for j in range(PCA_K):
        top5_idx = np.argsort(np.abs(avg_comps[j]))[::-1][:5]
        print(f"\\n  PC{j+1}:")
        for idx in top5_idx:
            sign = "+" if avg_comps[j][idx] >= 0 else "−"
            print(f"    {_short(_exog_names[idx]):<18}  {sign}{abs(avg_comps[j][idx]):.4f}")

# -- EN feature selection frequency ------------------------------------------
if _valid_coefs:
    en_coef_mat = np.array(_valid_coefs)   # (n_valid_steps, n_ar + k)
    n_steps = len(en_coef_mat)
    _ar_names  = [f"y_lag{i+1}" for i in range(N_AR_LAGS_EX)]
    _pca_names = [f"PC{i+1}"   for i in range(PCA_K)]
    _all_cand  = _ar_names + _pca_names

    print()
    print("Elastic Net feature selection (% of steps with non-zero coefficient):")
    print(f"  (Candidate pool: {len(_all_cand)} features — "
          f"{N_AR_LAGS_EX} AR lags + {PCA_K} PCA components)")
    print()
    print("  AR lags of target return:")
    for j, name in enumerate(_ar_names):
        pct = np.mean(np.abs(en_coef_mat[:, j]) > 1e-10)
        bar = "▪" * int(pct * 20)
        print(f"    {name:<10}  {pct:>6.1%}  {bar}")
    print()
    print("  PCA components:")
    for j, name in enumerate(_pca_names):
        pct = np.mean(np.abs(en_coef_mat[:, N_AR_LAGS_EX + j]) > 1e-10)
        bar = "▪" * int(pct * 20)
        print(f"    {name:<10}  {pct:>6.1%}  {bar}")

    avg_sel = np.mean(np.sum(np.abs(en_coef_mat) > 1e-10, axis=1))
    print(f"\\n  Average features selected per step: "
          f"{avg_sel:.1f} / {len(_all_cand)} candidates")

print("=" * 65)
""")

# ═══════════════════════════════════════════════════════════════════════════════
# 6.  Updated PLS backtest cell — adds diagnostic collection
# ═══════════════════════════════════════════════════════════════════════════════
PLS_BACKTEST = code("""\
# ═══════════════════════════════════════════════════════════════════════════════
# Model 2: ARIMAX-GARCH (PLS) — walk-forward backtest
# ═══════════════════════════════════════════════════════════════════════════════
# At each step:
#   1. PLS (c=3 components) on exog lag matrix supervised by y — training only
#   2. PLS X-scores combined with fixed AR lags (1..N_AR_FIXED)
#   3. OLS: y ~ [AR lags, PLS scores]
#   4. mu_hat = OLS.predict(test row)
#   5. residuals = y_train - OLS.predict(X_train)
#   6. GARCH(1,1) on residuals  →  one-step-ahead variance (rescale=False)
# Diagnostic data collected: PLS X-loadings, OLS coefficients.

pls_records = []
pls_fail    = 0

# Diagnostic collectors
_pls_loadings_all = []    # PLS X-loadings per step (n_exog, nc)
_pls_ols_coef_all = []    # OLS coefficients per step (N_AR_FIXED + nc,)
_pls_ols_intcp    = []    # OLS intercept per step

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

        _pls_loadings_all.append(pls.x_loadings_.copy())   # (n_exog, nc)

        # 2–3. OLS: y ~ [fixed AR lags, PLS scores]
        X_ols_tr = np.hstack([ar_tr, pls_sc_tr])
        X_ols_te = np.hstack([ar_te, pls_sc_te])
        ols = LinearRegression(fit_intercept=True)
        ols.fit(X_ols_tr, y_tr)

        _pls_ols_coef_all.append(ols.coef_.copy())
        _pls_ols_intcp.append(float(ols.intercept_))

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
        if len(_pls_loadings_all) < i + 1:
            _pls_loadings_all.append(None)
            _pls_ols_coef_all.append(None)
            _pls_ols_intcp.append(None)

    pls_records.append(_arimax_forecast_record(date, actual, mu_hat, sigma_hat))

pls_results = pd.DataFrame(pls_records).set_index("Date")

_lag_str_pls = ",".join(str(i) for i in range(1, N_AR_FIXED + 1))
PLS_NAME = f"ARX[{_lag_str_pls}]-GARCH(1,1) | PLS(c={N_COMP_PLS})"

print(f"\\nBacktest done. Failures carried forward: {pls_fail}/{TEST_SIZE}")
print(f"Model: {PLS_NAME}")
""")

# ═══════════════════════════════════════════════════════════════════════════════
# 7.  Updated PLS results cell — adds diagnostic printout
# ═══════════════════════════════════════════════════════════════════════════════
PLS_RESULTS = code("""\
# ── PLS: point-forecast metrics ───────────────────────────────────────────────
_pls_mse = float(np.mean((pls_results["actual"] - pls_results["pred_mean"])**2))
_pls_mae = float(np.mean(np.abs(pls_results["actual"] - pls_results["pred_mean"])))
_pls_dir = float(np.mean(np.sign(pls_results["actual"]) == np.sign(pls_results["pred_mean"])))

print(f"=== {PLS_NAME} ===")
print(f"MSE:           {_pls_mse:.8f}")
print(f"MAE:           {_pls_mae:.8f}")
print(f"Direction Acc: {_pls_dir:.2%}")
print()
display(pls_results[["actual","pred_mean","pred_std",
                       "ci_95_lower","ci_95_upper","var_95","var_99"]].head(5).round(6))

# ── PLS: diagnostic report ────────────────────────────────────────────────────
_valid_pls_ld  = [v for v in _pls_loadings_all if v is not None]
_valid_pls_co  = [v for v in _pls_ols_coef_all if v is not None]
_valid_pls_ic  = [v for v in _pls_ols_intcp    if v is not None]
_exog_names    = list(X_exog_dm.columns)

def _short(name):
    parts = name.split("_")
    return f"{parts[0]} lag{parts[-1]}"

print()
print("=" * 65)
print("PLS — DIAGNOSTIC REPORT")
print("=" * 65)
print(f"\\nExogenous block: {X_exog_dm.shape[1]} features  "
      f"({len(EXOG_INCLUDED)} sector ETFs × {N_EXOG_LAGS} lags)")
print(f"PLS extracts c = {N_COMP_PLS} supervised latent components.\\n")

# ETFs and lag inventory used
print("Sector ETFs entering the PLS compression stage:")
for t in EXOG_INCLUDED:
    lag_cols = [c for c in _exog_names if c.startswith(t + "_lag")]
    print(f"  {t:<8}  lags: {', '.join(c.split('_lag')[1] for c in lag_cols)}")

print()

# -- PLS X-loadings: top features per component --------------------------------
if _valid_pls_ld:
    avg_ld = np.mean(_valid_pls_ld, axis=0)   # (n_exog, nc)
    print("Top sector-lag contributors per PLS component (avg |X-loading|):")
    print("  (Higher loading = that sector-lag drives this component's direction)")
    for j in range(N_COMP_PLS):
        col_j = avg_ld[:, j]
        top5  = np.argsort(np.abs(col_j))[::-1][:5]
        print(f"\\n  PLS Component {j+1}:")
        for idx in top5:
            sign = "+" if col_j[idx] >= 0 else "−"
            print(f"    {_short(_exog_names[idx]):<18}  {sign}{abs(col_j[idx]):.4f}")

# -- OLS mean equation coefficients -------------------------------------------
if _valid_pls_co:
    avg_coef  = np.mean(_valid_pls_co, axis=0)
    avg_intcp = np.mean(_valid_pls_ic)
    _ar_lbl   = [f"y_lag{i+1}" for i in range(N_AR_FIXED)]
    _pl_lbl   = [f"PLS_score_{i+1}" for i in range(N_COMP_PLS)]
    _all_lbl  = _ar_lbl + _pl_lbl

    print()
    print("OLS mean equation — average coefficients across walk-forward steps:")
    print(f"  y_t = intercept + AR[1..{N_AR_FIXED}] + PLS_score_1 + ... + PLS_score_{N_COMP_PLS}")
    print()
    print(f"  {'Term':<18}  {'Avg Coeff':>12}")
    print(f"  {'-'*33}")
    print(f"  {'Intercept':<18}  {avg_intcp:>+12.6f}")
    for lbl, coef in zip(_all_lbl, avg_coef):
        print(f"  {lbl:<18}  {coef:>+12.6f}")

print("=" * 65)
""")

# ═══════════════════════════════════════════════════════════════════════════════
# 8.  Apply all content replacements
# ═══════════════════════════════════════════════════════════════════════════════

replacements = {
    "Model 1: ARIMAX-GARCH (PCA + Elastic Net) — walk-forward": PCA_EN_BACKTEST,
    "# ── PCA+EN results": PCA_EN_RESULTS,
    "Model 2: ARIMAX-GARCH (PLS) — walk-forward": PLS_BACKTEST,
    "# ── PLS results": PLS_RESULTS,
}

for marker, new_cell in replacements.items():
    idx = find_idx(marker)
    assert idx is not None, f"Could not find cell: {marker!r}"
    cells[idx]["source"] = new_cell["source"]
    print(f"[{idx}] Updated: {marker[:55]}")

# ═══════════════════════════════════════════════════════════════════════════════
# 9.  Insert new markdown cells (reverse order to keep indices stable)
# ═══════════════════════════════════════════════════════════════════════════════

# Find insertion anchor cells (after replacements, indices still stable)
idx_design  = find_idx("Step 3: Feature construction")          # after exog design matrix
idx_helpers = find_idx("ARIMAX configuration & shared helpers") # after ARIMAX helpers → PCA+EN md
idx_pca_res = find_idx("# ── PCA+EN: point-forecast metrics")   # after PCA+EN results → PLS md

assert idx_design  is not None, "Cannot find design matrix cell"
assert idx_helpers is not None, "Cannot find ARIMAX helpers cell"
assert idx_pca_res is not None, "Cannot find PCA+EN results cell"

print(f"\nInsertion anchors: design=[{idx_design}] helpers=[{idx_helpers}] pca_res=[{idx_pca_res}]")

# Insert in reverse positional order so earlier anchors stay valid
insertions = sorted(
    [(idx_pca_res, PLS_MD), (idx_helpers, PCA_EN_MD), (idx_design, EXOG_MD)],
    key=lambda x: x[0], reverse=True
)
for anchor, new_cell in insertions:
    cells.insert(anchor + 1, new_cell)
    print(f"  Inserted markdown after cell [{anchor}]")

# ═══════════════════════════════════════════════════════════════════════════════
# 10.  Write back
# ═══════════════════════════════════════════════════════════════════════════════
with open(NB_PATH, "w") as f:
    json.dump(nb, f, indent=1)

print(f"\nDone. Total cells: {len(cells)}")
print(f"✓ Notebook saved: {NB_PATH}")
