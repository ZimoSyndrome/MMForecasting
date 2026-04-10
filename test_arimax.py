"""
Integration test for the ARIMAX-GARCH pipeline.
Uses TEST_SIZE=20 for speed; full backtest is TEST_SIZE=126.

Run with:
    /Users/zimo/miniconda3/envs/bt3102/bin/python test_arimax.py
"""

import sys, warnings
sys.path.insert(0, "/Users/zimo/mmforecasting")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf

from src.exog_pipeline import (
    determine_exog_tickers,
    fetch_exog_returns,
    build_full_design_matrix,
    log_exog_config,
)
from src.arimax_models import (
    run_arimax_pca_en_backtest,
    run_arimax_pls_backtest,
    get_model_name_pca_en,
    get_model_name_pls,
)
from src.evaluation import compute_metrics, print_comparison_report

# ── Config ────────────────────────────────────────────────────────────────────
TICKER       = "SPY"
START_DATE   = "2012-01-01"
END_DATE     = "2025-12-31"
TEST_SIZE    = 20        # short for smoke-test; use 126 in notebook
PCA_K        = 3
N_COMP_PLS   = 3
N_AR_LAGS    = 5
N_EXOG_LAGS  = 5
N_AR_FIXED   = 3        # fixed AR lags used in PLS mean equation
DIST         = "t"

print(f"\n{'='*60}")
print(f"ARIMAX-GARCH Integration Test  |  {TICKER}  |  {TEST_SIZE} steps")
print(f"{'='*60}\n")

# ── 1. Target data ────────────────────────────────────────────────────────────
print("Fetching target data...")
raw = yf.download(TICKER, start=START_DATE, end=END_DATE,
                  progress=False, auto_adjust=True)
if isinstance(raw.columns, pd.MultiIndex):
    prices = raw["Close"].squeeze()
else:
    prices = raw["Close"]
target_returns = np.log(prices / prices.shift(1)).dropna()
target_returns.name = "Return"
print(f"  {len(target_returns)} obs  |  "
      f"{target_returns.index[0].date()} – {target_returns.index[-1].date()}")

# ── 2. Exog ticker selection ──────────────────────────────────────────────────
included, excluded, reasons = determine_exog_tickers(TICKER)

# ── 3. Fetch exog data ────────────────────────────────────────────────────────
print(f"\nFetching exog data ({len(included)} tickers)...")
exog_returns = fetch_exog_returns(included, START_DATE, END_DATE)
print(f"  Exog shape: {exog_returns.shape}")

# ── 4. Build design matrix ────────────────────────────────────────────────────
print("\nBuilding design matrix...")
y, X_ar, X_exog = build_full_design_matrix(
    target_returns, exog_returns, N_AR_LAGS, N_EXOG_LAGS
)
print(f"  Aligned obs : {len(y)}")
print(f"  X_ar shape  : {X_ar.shape}")
print(f"  X_exog shape: {X_exog.shape}")
print(f"  Effective start: {y.index[0].date()}")
print(f"  Test window   : {y.index[-(TEST_SIZE)].date()} – {y.index[-1].date()}")

log_exog_config(
    TICKER, included, excluded, reasons,
    PCA_K, N_COMP_PLS, N_AR_LAGS, N_EXOG_LAGS,
    design_shape=(len(y), X_ar.shape[1], X_exog.shape[1]),
    effective_start=str(y.index[0].date()),
)

# ── 5. ARIMAX-GARCH (PCA + EN) ────────────────────────────────────────────────
print("\n" + "─"*50)
pca_en_df, sel_history, sel_pca_history = run_arimax_pca_en_backtest(
    y, X_ar, X_exog,
    test_size=TEST_SIZE,
    pca_k=PCA_K,
    dist=DIST,
    n_ar_lags=N_AR_LAGS,
)
pca_en_name = get_model_name_pca_en(sel_history, sel_pca_history, PCA_K)
print(f"  Model: {pca_en_name}")
print(pca_en_df[["actual", "pred_mean", "pred_std",
                  "ci_95_lower", "ci_95_upper", "var_95", "var_99"]].round(6).head(5))

# ── 6. ARIMAX-GARCH (PLS) ─────────────────────────────────────────────────────
print("\n" + "─"*50)
pls_df = run_arimax_pls_backtest(
    y, X_ar, X_exog,
    test_size=TEST_SIZE,
    n_components=N_COMP_PLS,
    n_ar_fixed=N_AR_FIXED,
    dist=DIST,
)
pls_name = get_model_name_pls(N_AR_FIXED, N_COMP_PLS)
print(f"  Model: {pls_name}")
print(pls_df[["actual", "pred_mean", "pred_std",
               "ci_95_lower", "ci_95_upper", "var_95", "var_99"]].round(6).head(5))

# ── 7. Evaluation ─────────────────────────────────────────────────────────────
print("\n" + "─"*50)
prob_results = {pca_en_name: pca_en_df, pls_name: pls_df}
metrics_df = print_comparison_report(TICKER, TEST_SIZE, prob_results)
print("\nMetrics table:")
print(metrics_df.round(5))

print("\n✓ Integration test complete.\n")
