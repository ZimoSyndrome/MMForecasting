"""
Inject 9 cells (Part G: FRED Macro Extension) into the notebook
before the final Conclusions cell (currently index 74).

Cells added
-----------
  G0  (markdown) Part G section header + overview
  G1  (code)     FRED fetch + feature construction + log
  G2  (markdown) FRED methodology
  G3  (code)     Build combined exog matrix + macro design matrix
  G4  (code)     PCA+EN+FRED walk-forward backtest
  G5  (code)     PCA+EN+FRED results + diagnostics
  G6  (code)     PLS+FRED walk-forward backtest
  G7  (code)     PLS+FRED results + diagnostics
  G8  (code)     Macro-enhanced comparison report (4 ARIMAX models)
"""

import json

NB_PATH = "/Users/zimo/mmforecasting/notebooks/forecasting_analysis.ipynb"

with open(NB_PATH) as f:
    nb = json.load(f)

cells = nb["cells"]

# ── Find insertion point: just before Conclusions cell ────────────────────────
INSERT_IDX = None
for i, c in enumerate(cells):
    src = "".join(c["source"])
    if "Conclusions and Limitations" in src and c["cell_type"] == "markdown":
        INSERT_IDX = i
        break

if INSERT_IDX is None:
    # Fall back to appending at end
    INSERT_IDX = len(cells)
    print(f"Conclusions cell not found — appending at end ({INSERT_IDX})")
else:
    print(f"Inserting 9 cells before cell {INSERT_IDX} (Conclusions)")


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source,
    }


def md_cell(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# G0 — Section header
# ═══════════════════════════════════════════════════════════════════════════════
G0 = md_cell("""\
---
# Part G: FRED Macro Extension — Does a Small Macro Block Help?

This section tests whether adding **5 FRED macro-financial change variables**
to the existing sector-ETF exogenous matrix improves next-day return forecasting.

**FRED variables added:**

| FRED Code | Description | Transformation |
|---|---|---|
| DGS10 | 10-Year Treasury Constant Maturity Yield | Δ change |
| DGS2 | 2-Year Treasury Constant Maturity Yield | Δ change |
| DGS10 − DGS2 | Term Spread | Δ change |
| VIXCLS | CBOE VIX Index | Δ change |
| BAA10Y | Moody's BAA − 10Y Treasury (Credit Spread) | Δ change |

**Why changes, not levels?**
Yield and spread *levels* are non-stationary and would introduce spurious
correlations into a regression on stationary daily returns.  First-differencing
removes the level trend while preserving the economically relevant *shock*
information — a sudden spike in VIX or a sharp repricing of credit risk on
day *t* is a genuine information event for forecasting returns on day *t + 1*.

**Comparison objective:**
Run both ARX + PCA+EN and ARX + PLS on the *updated* exogenous matrix and
compare against the existing ETF-only results.\
""")


# ═══════════════════════════════════════════════════════════════════════════════
# G1 — FRED fetch + feature construction + log
# ═══════════════════════════════════════════════════════════════════════════════
G1 = code_cell("""\
# ── FRED macro-financial feature pipeline ────────────────────────────────────
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath("__file__")), "src"))
from fred_pipeline import (
    fetch_fred_raw, build_fred_features, log_fred_config,
    FRED_FEATURE_NAMES,
)

print("Fetching FRED macro data...")
fred_raw = fetch_fred_raw(start=START_DATE, end=END_DATE)

# Build first-difference features aligned to trading calendar (ETF index)
fred_features = build_fred_features(fred_raw, daily_index=exog_returns.index)

log_fred_config(fred_raw, fred_features)\
""")


# ═══════════════════════════════════════════════════════════════════════════════
# G2 — FRED methodology markdown
# ═══════════════════════════════════════════════════════════════════════════════
G2 = md_cell("""\
---
### FRED Macro Features: Construction Details

**Forward-fill rule (no look-ahead):**
FRED publishes Treasury yields and VIX daily on US business days.
On weekends and Fed holidays there is no new observation.  In the model,
the most recently available change is forward-filled for up to 5 calendar
days — this is equivalent to saying "no new macro news today, carry the
last known shock forward".  Same-day FRED releases are *not* used; the
lag structure in the design matrix (lag 1 = yesterday's change) ensures
all FRED information predates the forecast date.

**Feature lagging:**
Each of the 5 FRED change series is lagged 1 through 5 days, exactly as
the sector ETF returns are.  This gives 25 additional candidate features,
expanding the exogenous block from 55 (ETF-only) to 80 features:

| Block | Count |
|---|---|
| AR lags of target (y_lag1 … y_lag5) | 5 |
| Sector ETF return lags (11 ETFs × 5) | 55 |
| FRED macro change lags (5 vars × 5) | 25 |
| **Total candidates** | **85** |

PCA + Elastic Net and PLS are then applied to the full 80-column exogenous
block (unchanged from before), so the dimensionality reduction step absorbs
the macro variables automatically.\
""")


# ═══════════════════════════════════════════════════════════════════════════════
# G3 — Build combined exog matrix + macro design matrix
# ═══════════════════════════════════════════════════════════════════════════════
G3 = code_cell("""\
# ── Build combined exog matrix: ETF returns + FRED macro changes ──────────────
# Align FRED features to same trading-day index as ETF returns
fred_aligned = fred_features.reindex(exog_returns.index)

exog_returns_macro = pd.concat([exog_returns, fred_aligned], axis=1)

print(f"ETF exog block  : {exog_returns.shape[1]} columns")
print(f"FRED macro block: {fred_aligned.shape[1]} columns  "
      f"({', '.join(fred_aligned.columns.tolist())})")
print(f"Combined exog   : {exog_returns_macro.shape[1]} columns")

# Build design matrix with same lagging logic (reuse build_design_matrix from cell 12)
y_dm_m, X_ar_dm_m, X_exog_macro_dm = build_design_matrix(
    df["Return"], exog_returns_macro, N_AR_LAGS_EX, N_EXOG_LAGS
)
train_end_m = len(y_dm_m) - TEST_SIZE

n_exog_etf  = len(EXOG_INCLUDED) * N_EXOG_LAGS
n_exog_fred = len(fred_aligned.columns) * N_EXOG_LAGS

print()
print("=" * 60)
print("MACRO-ENHANCED DESIGN MATRIX")
print("=" * 60)
print(f"Aligned observations : {len(y_dm_m)}")
print(f"Effective start      : {y_dm_m.index[0].date()}")
print(f"Test window          : {y_dm_m.index[-TEST_SIZE].date()} – "
      f"{y_dm_m.index[-1].date()}")
print(f"AR features          : {X_ar_dm_m.shape[1]}")
print(f"Exog — ETF lags      : {n_exog_etf}  "
      f"({len(EXOG_INCLUDED)} ETFs × {N_EXOG_LAGS} lags)")
print(f"Exog — FRED lags     : {n_exog_fred}  "
      f"(5 macro vars × {N_EXOG_LAGS} lags)")
print(f"Exog — total         : {X_exog_macro_dm.shape[1]}")
print("=" * 60)\
""")


# ═══════════════════════════════════════════════════════════════════════════════
# G4 — PCA+EN+FRED walk-forward backtest
# ═══════════════════════════════════════════════════════════════════════════════
G4 = code_cell("""\
# ═══════════════════════════════════════════════════════════════════════════════
# Model 1+FRED: ARIMAX-GARCH (PCA + Elastic Net) on macro-enhanced exog matrix
# ═══════════════════════════════════════════════════════════════════════════════
pca_en_mac_records = []
pca_en_mac_fail    = 0
pca_en_mac_ar_hist = []   # selected AR lag indices per step
pca_en_mac_pc_hist = []   # selected PCA component indices per step

_pca_mac_var_ratios = []
_pca_mac_comps_all  = []
_en_mac_coefs_all   = []

_last_mu_pe_m  = 0.0
_last_sig_pe_m = float(np.std(y_dm_m.iloc[:train_end_m].values))

for i in tqdm(range(TEST_SIZE), desc="ARIMAX-GARCH PCA+EN+FRED"):
    actual = float(y_dm_m.iloc[train_end_m + i])
    date   = y_dm_m.index[train_end_m + i]

    y_tr  = y_dm_m.iloc[:train_end_m + i].values.astype(float)
    ar_tr = X_ar_dm_m.iloc[:train_end_m + i].values.astype(float)
    ex_tr = X_exog_macro_dm.iloc[:train_end_m + i].values.astype(float)
    ar_te = X_ar_dm_m.iloc[[train_end_m + i]].values.astype(float)
    ex_te = X_exog_macro_dm.iloc[[train_end_m + i]].values.astype(float)

    try:
        k = min(PCA_K, ex_tr.shape[1], ex_tr.shape[0] - 1)
        pca = PCA(n_components=k, random_state=RANDOM_SEED)
        fac_tr = pca.fit_transform(ex_tr)
        fac_te = pca.transform(ex_te)

        _pca_mac_var_ratios.append(pca.explained_variance_ratio_.copy())
        _pca_mac_comps_all.append(pca.components_.copy())

        X_cand_tr = np.hstack([ar_tr, fac_tr])
        X_cand_te = np.hstack([ar_te, fac_te])
        scaler_m = StandardScaler()
        X_sc_tr = scaler_m.fit_transform(X_cand_tr)
        X_sc_te = scaler_m.transform(X_cand_te)

        en = ElasticNetCV(l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 1.0],
                          cv=3, max_iter=10_000,
                          random_state=RANDOM_SEED, n_jobs=1)
        en.fit(X_sc_tr, y_tr)
        _en_mac_coefs_all.append(en.coef_.copy())

        mu_hat = float(en.predict(X_sc_te)[0])

        ar_coef = en.coef_[:N_AR_LAGS_EX]
        sel_ar  = [j + 1 for j, c in enumerate(ar_coef) if abs(c) > 1e-10]
        pca_en_mac_ar_hist.append(sel_ar)

        pc_coef = en.coef_[N_AR_LAGS_EX:N_AR_LAGS_EX + k]
        sel_pc  = [j + 1 for j, c in enumerate(pc_coef) if abs(c) > 1e-10]
        pca_en_mac_pc_hist.append(sel_pc)

        resid     = y_tr - en.predict(X_sc_tr)
        var_hat   = _fit_arimax_garch_var(resid, dist=DIST_ARIMAX)
        sigma_hat = float(np.sqrt(max(var_hat, 1e-12)))

        _last_mu_pe_m, _last_sig_pe_m = mu_hat, sigma_hat

    except Exception:
        pca_en_mac_fail += 1
        mu_hat, sigma_hat, sel_ar = _last_mu_pe_m, _last_sig_pe_m, []
        pca_en_mac_ar_hist.append(sel_ar)
        pca_en_mac_pc_hist.append([])
        if len(_pca_mac_var_ratios) < i + 1:
            _pca_mac_var_ratios.append(None)
            _pca_mac_comps_all.append(None)
            _en_mac_coefs_all.append(None)

    rec = _arimax_forecast_record(date, actual, mu_hat, sigma_hat)
    rec["selected_ar_lags"] = sel_ar
    pca_en_mac_records.append(rec)

pca_en_mac_results = pd.DataFrame(pca_en_mac_records).set_index("Date")

# Model name: frequent AR lags + PC components (>50% of steps)
_all_ar_m  = [l for lags in pca_en_mac_ar_hist for l in lags]
_cnt_ar_m  = Counter(_all_ar_m)
_freq_ar_m = sorted(l for l, c in _cnt_ar_m.items() if c > 0.5 * TEST_SIZE)

_all_pc_m  = [p for pcs in pca_en_mac_pc_hist for p in pcs]
_cnt_pc_m  = Counter(_all_pc_m)
_freq_pc_m = sorted(p for p, c in _cnt_pc_m.items() if c > 0.5 * TEST_SIZE)

_ar_str_m    = f"AR({','.join(str(l) for l in _freq_ar_m)})" if _freq_ar_m else ""
_pc_str_m    = f"X({','.join(f'PC{p}' for p in _freq_pc_m)})" if _freq_pc_m else ""
_mean_part_m = (_ar_str_m + _pc_str_m) if (_ar_str_m or _pc_str_m) else "intercept-only"
PCA_EN_MAC_NAME = f"{_mean_part_m}-GARCH(1,1) | PCA(k={PCA_K})+EN+FRED"

print(f"\\nBacktest done. Failures: {pca_en_mac_fail}/{TEST_SIZE}")
print(f"Model: {PCA_EN_MAC_NAME}")\
""")


# ═══════════════════════════════════════════════════════════════════════════════
# G5 — PCA+EN+FRED results + diagnostics
# ═══════════════════════════════════════════════════════════════════════════════
G5 = code_cell("""\
# ── PCA+EN+FRED: metrics + diagnostic report ─────────────────────────────────
print(f"=== {PCA_EN_MAC_NAME} ===")
_pe_m_mse = float(np.mean((pca_en_mac_results["actual"] - pca_en_mac_results["pred_mean"])**2))
_pe_m_mae = float(np.mean(np.abs(pca_en_mac_results["actual"] - pca_en_mac_results["pred_mean"])))
_pe_m_dir = float(np.mean(np.sign(pca_en_mac_results["actual"]) == np.sign(pca_en_mac_results["pred_mean"])))
print(f"MSE:           {_pe_m_mse:.8f}")
print(f"MAE:           {_pe_m_mae:.8f}")
print(f"Direction Acc: {_pe_m_dir:.2%}")
print()
display(pca_en_mac_results[["actual","pred_mean","pred_std",
                              "ci_95_lower","ci_95_upper","var_95","var_99"]].head(5).round(6))

# -- PCA variance explained ---------------------------------------------------
_valid_mac_var  = [v for v in _pca_mac_var_ratios if v is not None]
_valid_mac_comp = [c for c in _pca_mac_comps_all  if c is not None]
_valid_mac_coef = [c for c in _en_mac_coefs_all   if c is not None]
_mac_exog_names = list(X_exog_macro_dm.columns)

print()
print("=" * 65)
print("PCA+EN+FRED — DIAGNOSTIC REPORT")
print("=" * 65)

if _valid_mac_var:
    avg_var_m = np.mean(_valid_mac_var, axis=0)
    print(f"\\nExogenous block: {X_exog_macro_dm.shape[1]} features  "
          f"({len(EXOG_INCLUDED)} ETFs + 5 FRED vars) × {N_EXOG_LAGS} lags")
    print(f"PCA retains k = {PCA_K} components.\\n")
    print("Average variance explained (across all walk-forward steps):")
    print(f"  {'Component':<14} {'Var Explained':>14} {'Cumulative':>12}")
    cumvar = 0.0
    for j, v in enumerate(avg_var_m):
        cumvar += v
        bar = "█" * int(v * 40)
        print(f"  PC{j+1:<12}   {v:>10.1%}     {cumvar:>10.1%}   {bar}")

# -- PCA loadings: which features drive each component -----------------------
def _short_mac(name):
    parts = name.split("_lag", 1)
    return f"{parts[0]} lag{parts[1]}" if len(parts) == 2 else name

if _valid_mac_comp:
    avg_comps_m = np.mean(_valid_mac_comp, axis=0)   # (k, n_exog)
    print()
    print("Top feature exposures per PCA component (avg |loading|):")
    # Separate ETF vs FRED features in the top-5
    n_etf_feat  = len(EXOG_INCLUDED) * N_EXOG_LAGS
    for j in range(PCA_K):
        top5_idx = np.argsort(np.abs(avg_comps_m[j]))[::-1][:5]
        print(f"\\n  PC{j+1}:")
        for idx in top5_idx:
            sign = "+" if avg_comps_m[j][idx] >= 0 else "−"
            tag  = "[ETF]" if idx < n_etf_feat else "[FRED]"
            print(f"    {_short_mac(_mac_exog_names[idx]):<22}  {sign}{abs(avg_comps_m[j][idx]):.4f}  {tag}")

# -- EN selection frequency --------------------------------------------------
if _valid_mac_coef:
    en_coef_mat_m = np.array(_valid_mac_coef)
    n_steps_m = len(en_coef_mat_m)
    _ar_names_m  = [f"y_lag{i+1}" for i in range(N_AR_LAGS_EX)]
    _pca_names_m = [f"PC{i+1}"   for i in range(PCA_K)]
    _all_cand_m  = _ar_names_m + _pca_names_m

    print()
    print("Elastic Net feature selection (% of steps with non-zero coefficient):")
    print("  AR lags of target return:")
    for j, name in enumerate(_ar_names_m):
        pct = np.mean(np.abs(en_coef_mat_m[:, j]) > 1e-10)
        print(f"    {name:<10}  {pct:>6.1%}  {'▪' * int(pct * 20)}")
    print("  PCA components (now compress ETF + FRED jointly):")
    for j, name in enumerate(_pca_names_m):
        pct = np.mean(np.abs(en_coef_mat_m[:, N_AR_LAGS_EX + j]) > 1e-10)
        print(f"    {name:<10}  {pct:>6.1%}  {'▪' * int(pct * 20)}")
    avg_sel_m = np.mean(np.sum(np.abs(en_coef_mat_m) > 1e-10, axis=1))
    print(f"\\n  Avg features selected per step: {avg_sel_m:.1f} / {len(_all_cand_m)}")

print("=" * 65)\
""")


# ═══════════════════════════════════════════════════════════════════════════════
# G6 — PLS+FRED walk-forward backtest
# ═══════════════════════════════════════════════════════════════════════════════
G6 = code_cell("""\
# ═══════════════════════════════════════════════════════════════════════════════
# Model 2+FRED: ARIMAX-GARCH (PLS) on macro-enhanced exog matrix
# ═══════════════════════════════════════════════════════════════════════════════
pls_mac_records = []
pls_mac_fail    = 0

_pls_mac_loadings_all = []
_pls_mac_ols_coef_all = []
_pls_mac_ols_intcp    = []

_last_mu_pls_m  = 0.0
_last_sig_pls_m = float(np.std(y_dm_m.iloc[:train_end_m].values))

for i in tqdm(range(TEST_SIZE), desc="ARIMAX-GARCH PLS+FRED"):
    actual = float(y_dm_m.iloc[train_end_m + i])
    date   = y_dm_m.index[train_end_m + i]

    y_tr  = y_dm_m.iloc[:train_end_m + i].values.astype(float)
    ar_tr = X_ar_dm_m.iloc[:train_end_m + i, :N_AR_FIXED].values.astype(float)
    ex_tr = X_exog_macro_dm.iloc[:train_end_m + i].values.astype(float)
    ar_te = X_ar_dm_m.iloc[[train_end_m + i], :N_AR_FIXED].values.astype(float)
    ex_te = X_exog_macro_dm.iloc[[train_end_m + i]].values.astype(float)

    try:
        nc  = min(N_COMP_PLS, ex_tr.shape[1], ex_tr.shape[0] - 1)
        pls = PLSRegression(n_components=nc, scale=True)
        pls.fit(ex_tr, y_tr)
        pls_sc_tr = pls.transform(ex_tr)
        pls_sc_te = pls.transform(ex_te)

        _pls_mac_loadings_all.append(pls.x_loadings_.copy())

        X_ols_tr = np.hstack([ar_tr, pls_sc_tr])
        X_ols_te = np.hstack([ar_te, pls_sc_te])
        ols = LinearRegression(fit_intercept=True)
        ols.fit(X_ols_tr, y_tr)

        _pls_mac_ols_coef_all.append(ols.coef_.copy())
        _pls_mac_ols_intcp.append(float(ols.intercept_))

        mu_hat    = float(ols.predict(X_ols_te)[0])
        resid     = y_tr - ols.predict(X_ols_tr)
        var_hat   = _fit_arimax_garch_var(resid, dist=DIST_ARIMAX)
        sigma_hat = float(np.sqrt(max(var_hat, 1e-12)))

        _last_mu_pls_m, _last_sig_pls_m = mu_hat, sigma_hat

    except Exception:
        pls_mac_fail += 1
        mu_hat, sigma_hat = _last_mu_pls_m, _last_sig_pls_m
        if len(_pls_mac_loadings_all) < i + 1:
            _pls_mac_loadings_all.append(None)
            _pls_mac_ols_coef_all.append(None)
            _pls_mac_ols_intcp.append(None)

    pls_mac_records.append(_arimax_forecast_record(date, actual, mu_hat, sigma_hat))

pls_mac_results = pd.DataFrame(pls_mac_records).set_index("Date")
PLS_MAC_NAME = f"ARX[1,2,3]X(PLS{N_COMP_PLS})-GARCH(1,1) | PLS(c={N_COMP_PLS})+FRED"

print(f"\\nBacktest done. Failures: {pls_mac_fail}/{TEST_SIZE}")
print(f"Model: {PLS_MAC_NAME}")\
""")


# ═══════════════════════════════════════════════════════════════════════════════
# G7 — PLS+FRED results + diagnostics
# ═══════════════════════════════════════════════════════════════════════════════
G7 = code_cell("""\
# ── PLS+FRED: metrics + diagnostic report ────────────────────────────────────
print(f"=== {PLS_MAC_NAME} ===")
_pls_m_mse = float(np.mean((pls_mac_results["actual"] - pls_mac_results["pred_mean"])**2))
_pls_m_mae = float(np.mean(np.abs(pls_mac_results["actual"] - pls_mac_results["pred_mean"])))
_pls_m_dir = float(np.mean(np.sign(pls_mac_results["actual"]) == np.sign(pls_mac_results["pred_mean"])))
print(f"MSE:           {_pls_m_mse:.8f}")
print(f"MAE:           {_pls_m_mae:.8f}")
print(f"Direction Acc: {_pls_m_dir:.2%}")
print()
display(pls_mac_results[["actual","pred_mean","pred_std",
                           "ci_95_lower","ci_95_upper","var_95","var_99"]].head(5).round(6))

_valid_pls_m_ld = [v for v in _pls_mac_loadings_all if v is not None]
_valid_pls_m_co = [v for v in _pls_mac_ols_coef_all if v is not None]
_valid_pls_m_ic = [v for v in _pls_mac_ols_intcp    if v is not None]
_mac_exog_names = list(X_exog_macro_dm.columns)

print()
print("=" * 65)
print("PLS+FRED — DIAGNOSTIC REPORT")
print("=" * 65)

n_etf_feat_pls  = len(EXOG_INCLUDED) * N_EXOG_LAGS
n_fred_feat_pls = len(list(FRED_FEATURE_NAMES.keys())) * N_EXOG_LAGS

print(f"\\nExogenous block: {X_exog_macro_dm.shape[1]} features  "
      f"(ETF: {n_etf_feat_pls} + FRED: {n_fred_feat_pls})")
print(f"PLS extracts c = {N_COMP_PLS} supervised latent components.\\n")

# -- PLS X-loadings -----------------------------------------------------------
def _short_mac(name):
    parts = name.split("_lag", 1)
    return f"{parts[0]} lag{parts[1]}" if len(parts) == 2 else name

if _valid_pls_m_ld:
    avg_ld_m = np.mean(_valid_pls_m_ld, axis=0)   # (n_exog, nc)
    print("Top feature contributors per PLS component (avg |X-loading|):")
    for j in range(N_COMP_PLS):
        col_j = avg_ld_m[:, j]
        top5  = np.argsort(np.abs(col_j))[::-1][:5]
        print(f"\\n  PLS Component {j+1}:")
        for idx in top5:
            sign = "+" if col_j[idx] >= 0 else "−"
            tag  = "[ETF]" if idx < n_etf_feat_pls else "[FRED]"
            print(f"    {_short_mac(_mac_exog_names[idx]):<22}  {sign}{abs(col_j[idx]):.4f}  {tag}")

# -- OLS mean equation --------------------------------------------------------
if _valid_pls_m_co:
    avg_coef_m  = np.mean(_valid_pls_m_co, axis=0)
    avg_intcp_m = np.mean(_valid_pls_m_ic)
    _ar_lbl_m   = [f"y_lag{i+1}" for i in range(N_AR_FIXED)]
    _pl_lbl_m   = [f"PLS_score_{i+1}" for i in range(N_COMP_PLS)]
    print()
    print(f"OLS mean equation — average coefficients across walk-forward steps:")
    print(f"  {'Term':<18}  {'Avg Coeff':>12}")
    print(f"  {'-'*33}")
    print(f"  {'Intercept':<18}  {avg_intcp_m:>+12.6f}")
    for lbl, coef in zip(_ar_lbl_m + _pl_lbl_m, avg_coef_m):
        print(f"  {lbl:<18}  {coef:>+12.6f}")

print("=" * 65)\
""")


# ═══════════════════════════════════════════════════════════════════════════════
# G8 — Macro-enhanced comparison report (4 ARIMAX models)
# ═══════════════════════════════════════════════════════════════════════════════
G8 = code_cell("""\
# ── Macro-enhanced comparison: ETF-only vs ETF+FRED ──────────────────────────
macro_models = {
    PCA_EN_NAME:     pca_en_results,
    PCA_EN_MAC_NAME: pca_en_mac_results,
    PLS_NAME:        pls_results,
    PLS_MAC_NAME:    pls_mac_results,
}

macro_metrics = {name: compute_full_metrics(df_) for name, df_ in macro_models.items()}

W = 80
print()
print("=" * W)
print(f"MACRO EXTENSION COMPARISON REPORT: {TICKER}")
print("=" * W)
print(f"\\nBacktest: {TEST_SIZE} days walk-forward | "
      f"ETF-only exog (55 feat)  vs  ETF+FRED exog (80 feat)\\n")

_hdr = (f"{'Model':<46} {'MSE':>9} {'Dir Acc':>9} "
        f"{'CI Cov':>9} {'VaR Br':>8}")
print(_hdr)
print("-" * W)

for name, m in macro_metrics.items():
    tag = " ← +FRED" if "FRED" in name else ""
    print(f"{name:<46} "
          f"{m['MSE']:>9.5f} "
          f"{m['Dir Acc']:>8.1%} "
          f"{m['CI Coverage (95%)']:>8.1%} "
          f"{m['VaR Breach (95%)']:>7.1%}"
          f"{tag}")

print("-" * W)

# Delta rows: FRED improvement over ETF-only baseline
print("\\nΔ FRED improvement (positive = better for Dir Acc / CI Cov; "
      "negative = better for MSE / VaR Breach gap from 5%):")

for (base_name, base_m), (mac_name, mac_m) in [
    ((PCA_EN_NAME, macro_metrics[PCA_EN_NAME]),
     (PCA_EN_MAC_NAME, macro_metrics[PCA_EN_MAC_NAME])),
    ((PLS_NAME, macro_metrics[PLS_NAME]),
     (PLS_MAC_NAME, macro_metrics[PLS_MAC_NAME])),
]:
    d_mse  = mac_m["MSE"]    - base_m["MSE"]
    d_dir  = mac_m["Dir Acc"] - base_m["Dir Acc"]
    d_ci   = mac_m["CI Coverage (95%)"] - base_m["CI Coverage (95%)"]
    d_var  = mac_m["VaR Breach (95%)"]  - base_m["VaR Breach (95%)"]
    print(f"\\n  {base_name} → +FRED")
    print(f"    ΔMSE:         {d_mse:>+.6f}  {'↑ worse' if d_mse>0 else '↓ better'}")
    print(f"    ΔDir Acc:     {d_dir:>+.1%}  {'↑ better' if d_dir>0 else '↓ worse'}")
    print(f"    ΔCI Coverage: {d_ci:>+.1%}  (target 95%)")
    print(f"    ΔVaR Breach:  {d_var:>+.1%}  (target 5%)")

print()
print("=" * W)
print()
print("Interpretation guide:")
print("  MSE     : lower is better for point accuracy")
print("  Dir Acc : higher is better (50% = random baseline)")
print("  CI Cov  : should be ≈ 95% — too high = overly wide bands")
print("  VaR Br  : should be ≈ 5%  — too low = overly conservative VaR")
print()

# Quick verdict
print("Verdict:")
for (base_name, base_m), (mac_name, mac_m) in [
    ((PCA_EN_NAME, macro_metrics[PCA_EN_NAME]),
     (PCA_EN_MAC_NAME, macro_metrics[PCA_EN_MAC_NAME])),
    ((PLS_NAME, macro_metrics[PLS_NAME]),
     (PLS_MAC_NAME, macro_metrics[PLS_MAC_NAME])),
]:
    gains = []
    if mac_m["MSE"] < base_m["MSE"]:
        gains.append("MSE ↓")
    if mac_m["Dir Acc"] > base_m["Dir Acc"]:
        gains.append("Dir ↑")
    base_name_short = "PCA+EN" if "PCA" in base_name else "PLS"
    if gains:
        print(f"  {base_name_short} + FRED improves on: {', '.join(gains)}")
    else:
        print(f"  {base_name_short} + FRED shows no improvement on point-forecast metrics")
print("=" * W)\
""")


# ── Assemble and insert cells ─────────────────────────────────────────────────
new_cells = [G0, G1, G2, G3, G4, G5, G6, G7, G8]

cells[INSERT_IDX:INSERT_IDX] = new_cells

print(f"\\nInserted {len(new_cells)} cells at index {INSERT_IDX}")
print(f"Notebook now has {len(cells)} cells")

with open(NB_PATH, "w") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("✓ Notebook saved.")
