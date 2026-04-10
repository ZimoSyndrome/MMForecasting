"""
ARIMAX-GARCH models for next-day return forecasting.

Model 1: ARIMAX-GARCH (PCA + Elastic Net)
  - PCA on exogenous lag block (fit on train window only)
  - Elastic Net selects among [AR lags 1-5 of target] + [PCA factors]
  - GARCH(1,1) on mean-model residuals

Model 2: ARIMAX-GARCH (PLS)
  - PLS on exogenous lag block supervised by y_train
  - OLS: y ~ fixed AR lags + PLS scores
  - GARCH(1,1) on OLS residuals

Both models output full probabilistic forecasts:
  pred_mean, pred_variance, pred_std, ci_95_lower, ci_95_upper, var_95, var_99
"""

import numpy as np
import pandas as pd
from collections import Counter
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import ElasticNetCV, LinearRegression
from sklearn.preprocessing import StandardScaler
from arch import arch_model
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def build_forecast_record(date, actual: float, mu: float, sigma: float) -> dict:
    """
    Build a complete probabilistic forecast record.

    Definitions (per spec):
      CI 95%  : mu ± 1.96 * sigma
      VaR_95  : -(mu + z_{0.05} * sigma)  = -mu + 1.645 * sigma   [>0 = max expected loss]
      VaR_99  : -(mu + z_{0.01} * sigma)  = -mu + 2.326 * sigma
    """
    ci_lower = mu - 1.96 * sigma
    ci_upper = mu + 1.96 * sigma
    var_95 = -(mu + stats.norm.ppf(0.05) * sigma)   # = -mu + 1.6449*sigma
    var_99 = -(mu + stats.norm.ppf(0.01) * sigma)   # = -mu + 2.3263*sigma

    return {
        "Date": date,
        "actual": float(actual),
        "pred_mean": float(mu),
        "pred_variance": float(sigma ** 2),
        "pred_std": float(sigma),
        "ci_95_lower": float(ci_lower),
        "ci_95_upper": float(ci_upper),
        "var_95": float(var_95),
        "var_99": float(var_99),
    }


def _fit_garch_variance(residuals: np.ndarray, dist: str = "t") -> float:
    """
    Fit GARCH(1,1) with Zero mean on the supplied residuals.
    Returns the one-step-ahead conditional variance in the ORIGINAL scale.

    arch 6+ auto-rescales data by `model.scale` for numerical stability;
    `forecast().variance` is returned in the RESCALED units, so we must
    divide by scale² to recover the original-scale variance.
    Using rescale=False avoids this entirely and is safe for daily returns.
    """
    am = arch_model(
        residuals, mean="Zero", vol="Garch", p=1, q=1, dist=dist, rescale=False
    )
    gfit = am.fit(disp="off", show_warning=False)
    fcast = gfit.forecast(horizon=1, reindex=False)
    var_hat = float(fcast.variance.iloc[-1, 0])
    return var_hat


# ---------------------------------------------------------------------------
# Model 1: ARIMAX-GARCH (PCA + Elastic Net)
# ---------------------------------------------------------------------------

def run_arimax_pca_en_backtest(
    y: pd.Series,
    X_ar: pd.DataFrame,
    X_exog: pd.DataFrame,
    test_size: int,
    pca_k: int = 3,
    dist: str = "t",
    n_ar_lags: int = 5,
    random_seed: int = 42,
) -> tuple:
    """
    Walk-forward expanding-window backtest for ARIMAX-GARCH (PCA + Elastic Net).

    At each step i ∈ [0, test_size):
      1. Fit PCA (k components) on exog lag matrix using training rows only.
      2. Build candidate matrix X = [AR lags 1-5, PCA factors 1-k].
      3. StandardScaler on X (fit on train only).
      4. ElasticNetCV selects sparse subset of predictors.
      5. Mean forecast: mu_hat = EN.predict(X_test_row).
      6. Training residuals  = y_train - EN.predict(X_train).
      7. Fit GARCH(1,1) on residuals → one-step-ahead variance forecast.
      8. sigma_hat = sqrt(var_hat).

    Returns
    -------
    forecast_df            : pd.DataFrame with full probabilistic forecast columns
    sel_ar_lags_history    : list of lists — selected AR lag indices per step
    """
    train_end = len(y) - test_size
    records = []
    fail_count = 0
    sel_ar_lags_history = []
    sel_pca_history = []

    # Sensible carry-forward defaults
    last_mu = 0.0
    last_sigma = float(np.std(y.iloc[:train_end].values))
    last_sel_ar = []

    for i in tqdm(range(test_size), desc="ARIMAX-GARCH (PCA+EN) walk-forward"):
        actual = float(y.iloc[train_end + i])
        date = y.index[train_end + i]

        # --- Training slices --------------------------------------------------
        y_tr   = y.iloc[: train_end + i].values.astype(float)
        ar_tr  = X_ar.iloc[: train_end + i].values.astype(float)
        ex_tr  = X_exog.iloc[: train_end + i].values.astype(float)

        # --- Test row (single observation) ------------------------------------
        ar_te  = X_ar.iloc[[train_end + i]].values.astype(float)   # (1, n_ar)
        ex_te  = X_exog.iloc[[train_end + i]].values.astype(float) # (1, n_exog_feats)

        try:
            # 1. PCA on exog block — fit on TRAINING data only
            k = min(pca_k, ex_tr.shape[1], ex_tr.shape[0] - 1)
            pca = PCA(n_components=k, random_state=random_seed)
            factors_tr = pca.fit_transform(ex_tr)   # (n_train, k)
            factors_te = pca.transform(ex_te)        # (1, k)

            # 2. Candidate matrix: AR lags + PCA factors
            X_cand_tr = np.hstack([ar_tr, factors_tr])   # (n_train, n_ar + k)
            X_cand_te = np.hstack([ar_te, factors_te])   # (1, n_ar + k)

            # 3. StandardScaler (fit on train, transform both)
            scaler = StandardScaler()
            X_sc_tr = scaler.fit_transform(X_cand_tr)
            X_sc_te = scaler.transform(X_cand_te)

            # 4. ElasticNetCV — sparse selection over candidate predictors
            #    (Elastic Net performs lag selection, NOT classical ARIMA order
            #     estimation. It finds which AR lags and PCA factors matter.)
            en = ElasticNetCV(
                l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 1.0],
                cv=3,
                max_iter=10_000,
                random_state=random_seed,
                n_jobs=1,
            )
            en.fit(X_sc_tr, y_tr)

            # 5. Mean forecast
            mu_hat = float(en.predict(X_sc_te)[0])

            # Track selected AR lags (first n_ar_lags coefficients in X_cand)
            ar_coef = en.coef_[:n_ar_lags]
            sel_ar = [j + 1 for j, c in enumerate(ar_coef) if abs(c) > 1e-10]
            sel_ar_lags_history.append(sel_ar)
            last_sel_ar = sel_ar

            # Track selected PCA components (next k coefficients in X_cand)
            pc_coef = en.coef_[n_ar_lags:n_ar_lags + k]
            sel_pca = [j + 1 for j, c in enumerate(pc_coef) if abs(c) > 1e-10]
            sel_pca_history.append(sel_pca)

            # 6. Training residuals
            mu_tr_hat = en.predict(X_sc_tr)
            residuals = y_tr - mu_tr_hat

            # 7. GARCH(1,1) on residuals
            var_hat = _fit_garch_variance(residuals, dist=dist)
            sigma_hat = float(np.sqrt(max(var_hat, 1e-12)))

            last_mu, last_sigma = mu_hat, sigma_hat

        except Exception:
            fail_count += 1
            mu_hat, sigma_hat = last_mu, last_sigma
            sel_ar = last_sel_ar
            sel_ar_lags_history.append(sel_ar)
            sel_pca_history.append([])

        record = build_forecast_record(date, actual, mu_hat, sigma_hat)
        record["selected_ar_lags"] = sel_ar
        records.append(record)

    print(f"  PCA+EN backtest done — failures carried forward: {fail_count}/{test_size}")
    df = pd.DataFrame(records).set_index("Date")
    df["pca_k"] = pca_k
    return df, sel_ar_lags_history, sel_pca_history


# ---------------------------------------------------------------------------
# Model 2: ARIMAX-GARCH (PLS)
# ---------------------------------------------------------------------------

def run_arimax_pls_backtest(
    y: pd.Series,
    X_ar: pd.DataFrame,
    X_exog: pd.DataFrame,
    test_size: int,
    n_components: int = 3,
    n_ar_fixed: int = 3,
    dist: str = "t",
) -> pd.DataFrame:
    """
    Walk-forward expanding-window backtest for ARIMAX-GARCH (PLS).

    At each step i ∈ [0, test_size):
      1. Fit PLS (n_components) on exog lag matrix supervised by y_train.
      2. Extract PLS scores for train + test row.
      3. OLS: y_train ~ [AR lags 1..n_ar_fixed, PLS scores].
      4. Mean forecast: mu_hat = OLS.predict(test row).
      5. Training residuals = y_train - OLS predictions.
      6. Fit GARCH(1,1) on residuals → one-step-ahead variance forecast.
      7. sigma_hat = sqrt(var_hat).

    n_ar_fixed : fixed (small) set of AR lags included in the mean equation.
                 PLS does not perform AR lag selection; a fixed candidate set
                 is used (lags 1 to n_ar_fixed).
    """
    train_end = len(y) - test_size
    records = []
    fail_count = 0

    last_mu = 0.0
    last_sigma = float(np.std(y.iloc[:train_end].values))

    for i in tqdm(range(test_size), desc="ARIMAX-GARCH (PLS) walk-forward"):
        actual = float(y.iloc[train_end + i])
        date = y.index[train_end + i]

        # Training slices (use only n_ar_fixed AR lag columns)
        y_tr   = y.iloc[: train_end + i].values.astype(float)
        ar_tr  = X_ar.iloc[: train_end + i, :n_ar_fixed].values.astype(float)
        ex_tr  = X_exog.iloc[: train_end + i].values.astype(float)

        ar_te  = X_ar.iloc[[train_end + i], :n_ar_fixed].values.astype(float)
        ex_te  = X_exog.iloc[[train_end + i]].values.astype(float)

        try:
            # 1. PLS on exog block supervised by y — fit on TRAINING data only
            nc = min(n_components, ex_tr.shape[1], ex_tr.shape[0] - 1)
            pls = PLSRegression(n_components=nc, scale=True)
            pls.fit(ex_tr, y_tr)

            pls_sc_tr = pls.transform(ex_tr)   # (n_train, nc)  — X scores
            pls_sc_te = pls.transform(ex_te)    # (1, nc)

            # 2. OLS: y ~ [fixed AR lags, PLS scores]
            X_ols_tr = np.hstack([ar_tr, pls_sc_tr])  # (n_train, n_ar_fixed + nc)
            X_ols_te = np.hstack([ar_te, pls_sc_te])  # (1, n_ar_fixed + nc)

            ols = LinearRegression(fit_intercept=True)
            ols.fit(X_ols_tr, y_tr)

            # 3. Mean forecast
            mu_hat = float(ols.predict(X_ols_te)[0])

            # 4. Training residuals
            mu_tr_hat = ols.predict(X_ols_tr)
            residuals = y_tr - mu_tr_hat

            # 5. GARCH(1,1)
            var_hat = _fit_garch_variance(residuals, dist=dist)
            sigma_hat = float(np.sqrt(max(var_hat, 1e-12)))

            last_mu, last_sigma = mu_hat, sigma_hat

        except Exception:
            fail_count += 1
            mu_hat, sigma_hat = last_mu, last_sigma

        records.append(build_forecast_record(date, actual, mu_hat, sigma_hat))

    print(f"  PLS backtest done — failures carried forward: {fail_count}/{test_size}")
    df = pd.DataFrame(records).set_index("Date")
    df["n_components"] = n_components
    df["n_ar_fixed"] = n_ar_fixed
    return df


# ---------------------------------------------------------------------------
# Model naming
# ---------------------------------------------------------------------------

def get_model_name_pca_en(
    sel_ar_lags_history: list, sel_pca_history: list, pca_k: int
) -> str:
    """
    Derive canonical model name from walk-forward AR lag and PCA component
    selection history.  Features that appeared in >50% of backtest steps are
    included in the name.

    Format examples:
      AR(1,3)X(PC1,PC4)-GARCH(1,1) | PCA(k=5)+EN   ← AR lags 1,3 + PC1,PC4
      X(PC2,PC5)-GARCH(1,1) | PCA(k=5)+EN           ← no AR lags, only PCs
      AR(2)-GARCH(1,1) | PCA(k=5)+EN                ← only AR lag 2, no PCs
      intercept-only-GARCH(1,1) | PCA(k=5)+EN       ← EN zeroed everything
    """
    n_steps = max(len(sel_ar_lags_history), 1)

    # Frequent AR lags of target (>50% of steps)
    all_ar = [l for lags in sel_ar_lags_history for l in lags]
    cnt_ar = Counter(all_ar)
    freq_ar = sorted(l for l, c in cnt_ar.items() if c > 0.5 * n_steps)

    # Frequent PCA components (>50% of steps)
    all_pc = [p for pcs in sel_pca_history for p in pcs]
    cnt_pc = Counter(all_pc)
    freq_pc = sorted(p for p, c in cnt_pc.items() if c > 0.5 * n_steps)

    ar_str = f"AR({','.join(str(l) for l in freq_ar)})" if freq_ar else ""
    pc_str = f"X({','.join(f'PC{p}' for p in freq_pc)})" if freq_pc else ""
    mean_part = (ar_str + pc_str) if (ar_str or pc_str) else "intercept-only"

    return f"{mean_part}-GARCH(1,1) | PCA(k={pca_k})+EN"


def get_model_name_pls(n_ar_fixed: int, n_components: int) -> str:
    """Canonical model name for the PLS variant."""
    lag_str = ",".join(str(i) for i in range(1, n_ar_fixed + 1))
    return f"ARX[{lag_str}]-GARCH(1,1) | PLS(c={n_components})"
