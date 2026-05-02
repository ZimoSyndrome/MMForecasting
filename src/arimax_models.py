"""
ARIMAX-GARCH models for next-day return forecasting.

Both models fit the mean equation by **joint maximum likelihood** via
`statsmodels.tsa.arima.model.ARIMA` with the compressed exogenous block as
`exog`. A GARCH(1,1) on the mean-equation residuals supplies the
conditional-variance step. The `(p, q)` order is chosen once on the initial
training window by AIC grid-search with the exogenous block already present
(no omitted-variable bias), then held fixed through the walk-forward loop.

Model 1: ARIMAX-GARCH (PCA)
  - PCA on the exogenous lag block (fit on the training window only).
  - ARIMA(y, exog=PCA_scores, order=(p, 0, q)).fit() does the joint MLE.
  - GARCH(1,1) on the ARIMAX residuals.

Model 2: ARIMAX-GARCH (PLS)
  - PLS on the exogenous lag block supervised by y_train.
  - ARIMA(y, exog=PLS_scores, order=(p, 0, q)).fit() does the joint MLE.
  - GARCH(1,1) on the ARIMAX residuals.

Both models output full probabilistic forecasts:
  pred_mean, pred_variance, pred_std, ci_95_lower, ci_95_upper, var_95, var_99
"""

import warnings

import numpy as np
import pandas as pd
from arch import arch_model
from scipy import stats
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from statsmodels.tsa.arima.model import ARIMA
from tqdm.auto import tqdm

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
    var_95 = -(mu + stats.norm.ppf(0.05) * sigma)
    var_99 = -(mu + stats.norm.ppf(0.01) * sigma)

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
    `rescale=False` avoids arch's automatic rescaling (which returns the
    forecast variance in rescaled units).
    """
    am = arch_model(
        residuals, mean="Zero", vol="Garch", p=1, q=1, dist=dist, rescale=False
    )
    gfit = am.fit(disp="off", show_warning=False)
    fcast = gfit.forecast(horizon=1, reindex=False)
    return float(fcast.variance.iloc[-1, 0])


def select_arimax_order(
    y: np.ndarray,
    X_exog: np.ndarray,
    p_range,
    q_range,
) -> tuple:
    """
    One-off AIC grid-search for ARIMAX(p, 0, q) with given exogenous block.

    Fits ARIMA(y, exog=X_exog, order=(p, 0, q), trend='c') by joint MLE for
    every (p, q) in the Cartesian product of p_range × q_range. Returns the
    (p, q) with the lowest AIC. Ties are broken by parsimony (smaller p
    first, then smaller q). Fits that fail to converge are silently
    skipped. If every fit fails, falls back to (0, 0) with a warning.
    """
    best = None  # (aic, p, q)
    for p in p_range:
        for q in q_range:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    fit = ARIMA(
                        y, exog=X_exog, order=(p, 0, q), trend="c"
                    ).fit()
                aic = float(fit.aic)
                if not np.isfinite(aic):
                    continue
                cand = (aic, p, q)
                if best is None or cand < best:
                    best = cand
            except Exception:
                continue
    if best is None:
        warnings.warn(
            "ARIMAX order grid-search: all fits failed. Falling back to (0, 0)."
        )
        return 0, 0
    _, best_p, best_q = best
    return best_p, best_q


def _extract_params(mean_fit, p: int, q: int, k: int) -> dict:
    """Extract (const, ar_1..ar_p, ma_1..ma_q, β_1..β_k) from an ARIMA fit."""
    params = mean_fit.params
    return {
        "const": float(params.get("const", 0.0)),
        "ar": [float(params.get(f"ar.L{j + 1}", 0.0)) for j in range(p)],
        "ma": [float(params.get(f"ma.L{j + 1}", 0.0)) for j in range(q)],
        "exog": [float(params.get(f"x{j + 1}", 0.0)) for j in range(k)],
    }


# ---------------------------------------------------------------------------
# Model 1: ARIMAX-GARCH (PCA)
# ---------------------------------------------------------------------------

def run_arimax_pca_backtest(
    y: pd.Series,
    X_exog: pd.DataFrame,
    test_size: int,
    pca_k: int = 5,
    max_ar: int = 4,
    max_ma: int = 2,
    dist: str = "t",
    random_seed: int = 42,
) -> tuple:
    """
    Walk-forward backtest for ARIMAX-GARCH with a PCA-compressed exog block.

    One-off AIC (p, q) grid-search is run on the initial training window with
    PCA scores already present as the `exog`.  The chosen `(p, q)` is then
    held fixed through the walk-forward loop.  Each step refits PCA (fit on
    the expanding training slice only), fits `ARIMA(y, exog=PCA_scores,
    order=(p, 0, q), trend='c')` by joint MLE, forecasts one step, and fits
    GARCH(1,1) on the mean-equation residuals for σ̂.

    Returns
    -------
    forecast_df : pd.DataFrame with full probabilistic forecast columns.
    diagnostics : dict with keys {'best_p', 'best_q', 'pca_var_ratios',
                                  'pca_components', 'mean_params'}.
    """
    train_end = len(y) - test_size
    records = []
    fail_count = 0

    pca_var_ratios: list = []
    pca_components_all: list = []
    mean_params_all: list = []

    last_mu = 0.0
    last_sigma = float(np.std(y.iloc[:train_end].values))

    # One-off (p, q) AIC grid-search with PCA exog on the initial window.
    k_init = min(pca_k, X_exog.shape[1], train_end - 1)
    pca_init = PCA(n_components=k_init, random_state=random_seed)
    ex_init = X_exog.iloc[:train_end].values.astype(float)
    y_init = y.iloc[:train_end].values.astype(float)
    fac_init = pca_init.fit_transform(ex_init)

    best_p, best_q = select_arimax_order(
        y_init, fac_init, range(0, max_ar + 1), range(0, max_ma + 1)
    )
    print(
        f"  ARIMAX(PCA) order chosen by AIC on initial window: "
        f"(p={best_p}, d=0, q={best_q})"
    )

    for i in tqdm(range(test_size), desc="ARIMAX-GARCH (PCA) walk-forward"):
        actual = float(y.iloc[train_end + i])
        date = y.index[train_end + i]

        y_tr = y.iloc[: train_end + i].values.astype(float)
        ex_tr = X_exog.iloc[: train_end + i].values.astype(float)
        ex_te = X_exog.iloc[[train_end + i]].values.astype(float)

        try:
            k = min(pca_k, ex_tr.shape[1], ex_tr.shape[0] - 1)
            pca = PCA(n_components=k, random_state=random_seed)
            fac_tr = pca.fit_transform(ex_tr)
            fac_te = pca.transform(ex_te)

            pca_var_ratios.append(pca.explained_variance_ratio_.copy())
            pca_components_all.append(pca.components_.copy())

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mean_fit = ARIMA(
                    y_tr, exog=fac_tr, order=(best_p, 0, best_q), trend="c"
                ).fit()

            mu_hat = float(mean_fit.forecast(steps=1, exog=fac_te).iloc[0])
            mean_params_all.append(_extract_params(mean_fit, best_p, best_q, k))

            residuals = np.asarray(mean_fit.resid, dtype=float)
            var_hat = _fit_garch_variance(residuals, dist=dist)
            sigma_hat = float(np.sqrt(max(var_hat, 1e-12)))

            last_mu, last_sigma = mu_hat, sigma_hat

        except Exception:
            fail_count += 1
            mu_hat, sigma_hat = last_mu, last_sigma
            if len(pca_var_ratios) < i + 1:
                pca_var_ratios.append(None)
                pca_components_all.append(None)
                mean_params_all.append(None)

        records.append(build_forecast_record(date, actual, mu_hat, sigma_hat))

    print(
        f"  PCA-ARIMAX backtest done. Failures carried forward: "
        f"{fail_count}/{test_size}"
    )
    df = pd.DataFrame(records).set_index("Date")
    df["pca_k"] = pca_k
    diagnostics = {
        "best_p": best_p,
        "best_q": best_q,
        "pca_var_ratios": pca_var_ratios,
        "pca_components": pca_components_all,
        "mean_params": mean_params_all,
    }
    return df, diagnostics


# ---------------------------------------------------------------------------
# Model 2: ARIMAX-GARCH (PLS)
# ---------------------------------------------------------------------------

def run_arimax_pls_backtest(
    y: pd.Series,
    X_exog: pd.DataFrame,
    test_size: int,
    n_components: int = 3,
    max_ar: int = 4,
    max_ma: int = 2,
    dist: str = "t",
) -> tuple:
    """
    Walk-forward backtest for ARIMAX-GARCH with a PLS-compressed exog block.

    Structure mirrors `run_arimax_pca_backtest`: one-off (p, q) AIC
    grid-search on the initial window with PLS scores as `exog`,
    per-window joint ARIMAX MLE, GARCH(1,1) on residuals.

    Returns
    -------
    forecast_df : pd.DataFrame with full probabilistic forecast columns.
    diagnostics : dict with keys {'best_p', 'best_q', 'pls_x_loadings',
                                  'mean_params'}.
    """
    train_end = len(y) - test_size
    records = []
    fail_count = 0

    pls_x_loadings: list = []
    mean_params_all: list = []

    last_mu = 0.0
    last_sigma = float(np.std(y.iloc[:train_end].values))

    # One-off (p, q) AIC grid-search with PLS exog on the initial window.
    c_init = min(n_components, X_exog.shape[1], train_end - 1)
    pls_init = PLSRegression(n_components=c_init, scale=True)
    ex_init = X_exog.iloc[:train_end].values.astype(float)
    y_init = y.iloc[:train_end].values.astype(float)
    pls_init.fit(ex_init, y_init)
    sc_init = pls_init.transform(ex_init)

    best_p, best_q = select_arimax_order(
        y_init, sc_init, range(0, max_ar + 1), range(0, max_ma + 1)
    )
    print(
        f"  ARIMAX(PLS) order chosen by AIC on initial window: "
        f"(p={best_p}, d=0, q={best_q})"
    )

    for i in tqdm(range(test_size), desc="ARIMAX-GARCH (PLS) walk-forward"):
        actual = float(y.iloc[train_end + i])
        date = y.index[train_end + i]

        y_tr = y.iloc[: train_end + i].values.astype(float)
        ex_tr = X_exog.iloc[: train_end + i].values.astype(float)
        ex_te = X_exog.iloc[[train_end + i]].values.astype(float)

        try:
            nc = min(n_components, ex_tr.shape[1], ex_tr.shape[0] - 1)
            pls = PLSRegression(n_components=nc, scale=True)
            pls.fit(ex_tr, y_tr)
            sc_tr = pls.transform(ex_tr)
            sc_te = pls.transform(ex_te)

            pls_x_loadings.append(pls.x_loadings_.copy())

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mean_fit = ARIMA(
                    y_tr, exog=sc_tr, order=(best_p, 0, best_q), trend="c"
                ).fit()

            mu_hat = float(mean_fit.forecast(steps=1, exog=sc_te).iloc[0])
            mean_params_all.append(_extract_params(mean_fit, best_p, best_q, nc))

            residuals = np.asarray(mean_fit.resid, dtype=float)
            var_hat = _fit_garch_variance(residuals, dist=dist)
            sigma_hat = float(np.sqrt(max(var_hat, 1e-12)))

            last_mu, last_sigma = mu_hat, sigma_hat

        except Exception:
            fail_count += 1
            mu_hat, sigma_hat = last_mu, last_sigma
            if len(pls_x_loadings) < i + 1:
                pls_x_loadings.append(None)
                mean_params_all.append(None)

        records.append(build_forecast_record(date, actual, mu_hat, sigma_hat))

    print(
        f"  PLS-ARIMAX backtest done. Failures carried forward: "
        f"{fail_count}/{test_size}"
    )
    df = pd.DataFrame(records).set_index("Date")
    df["n_components"] = n_components
    diagnostics = {
        "best_p": best_p,
        "best_q": best_q,
        "pls_x_loadings": pls_x_loadings,
        "mean_params": mean_params_all,
    }
    return df, diagnostics


# ---------------------------------------------------------------------------
# Model naming
# ---------------------------------------------------------------------------

def get_model_name_pca(best_p: int, best_q: int, pca_k: int) -> str:
    """Canonical model name for the PCA variant."""
    return f"ARIMAX({best_p},0,{best_q})-GARCH(1,1) | PCA(k={pca_k})"


def get_model_name_pls(best_p: int, best_q: int, n_components: int) -> str:
    """Canonical model name for the PLS variant."""
    return f"ARIMAX({best_p},0,{best_q})-GARCH(1,1) | PLS(c={n_components})"
