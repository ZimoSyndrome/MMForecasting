"""
Evaluation metrics and structured reporting for ARIMAX-GARCH models.

Metrics
-------
MSE            : mean squared error on point forecasts
MAE            : mean absolute error on point forecasts
Dir Acc        : % of steps where sign(pred) == sign(actual). 50% is random.
CI Coverage    : % of actuals inside predicted 95% CI. Target ~95%.
VaR Breach     : % of actuals that breach the 95% VaR. Target ~5%.
"""

import numpy as np
import pandas as pd


def compute_metrics(forecast_df: pd.DataFrame) -> dict:
    """
    Compute all evaluation metrics from a forecast DataFrame.

    Required columns:
        actual, pred_mean, pred_std, ci_95_lower, ci_95_upper, var_95
    """
    y  = forecast_df["actual"].values.astype(float)
    mu = forecast_df["pred_mean"].values.astype(float)
    ci_lo = forecast_df["ci_95_lower"].values.astype(float)
    ci_hi = forecast_df["ci_95_upper"].values.astype(float)
    v95   = forecast_df["var_95"].values.astype(float)

    n = len(y)
    if n == 0:
        raise ValueError("forecast_df is empty")

    mse      = float(np.mean((y - mu) ** 2))
    mae      = float(np.mean(np.abs(y - mu)))
    dir_acc  = float(np.mean(np.sign(y) == np.sign(mu)))

    # CI Coverage: actual inside [ci_lo, ci_hi]
    ci_cov   = float(np.mean((y >= ci_lo) & (y <= ci_hi)))

    # VaR Breach: actual loss exceeds VaR
    #   VaR_95 = -(mu + z_0.05*sigma) => breach when actual < -VaR_95 = mu - 1.645*sigma
    var_bch  = float(np.mean(y < -v95))

    return {
        "MSE":               mse,
        "MAE":               mae,
        "Dir Acc":           dir_acc,
        "CI Coverage (95%)": ci_cov,
        "VaR Breach (95%)":  var_bch,
    }


def compute_baseline_metrics(actual: np.ndarray, pred_mean: np.ndarray) -> dict:
    """
    Compute MSE, MAE, and directional accuracy for models that only have
    point forecasts (no CI / VaR). Fills missing metrics with NaN.
    """
    y  = np.asarray(actual, dtype=float)
    mu = np.asarray(pred_mean, dtype=float)
    return {
        "MSE":               float(np.mean((y - mu) ** 2)),
        "MAE":               float(np.mean(np.abs(y - mu))),
        "Dir Acc":           float(np.mean(np.sign(y) == np.sign(mu))),
        "CI Coverage (95%)": float("nan"),
        "VaR Breach (95%)":  float("nan"),
    }


def print_comparison_report(
    asset: str,
    test_size: int,
    prob_results: dict,           # {model_name: forecast_df with CI/VaR columns}
    baseline_metrics: dict = None, # {model_name: metrics_dict} for point-only models
) -> pd.DataFrame:
    """
    Print the structured MODEL COMPARISON REPORT and return a metrics DataFrame.

    Parameters
    ----------
    prob_results      : dict mapping model name → forecast DataFrame
                        (must have all probabilistic columns)
    baseline_metrics  : optional dict of pre-computed metrics for models
                        that only provide point forecasts (e.g. ARIMA-GARCH baseline,
                        XGBoost, LSTM)
    """
    all_metrics = {}

    # Probabilistic models (full metrics)
    for name, df in prob_results.items():
        all_metrics[name] = compute_metrics(df)

    # Point-forecast baselines (partial metrics)
    if baseline_metrics:
        for name, m in baseline_metrics.items():
            all_metrics[name] = m

    # --- Print report --------------------------------------------------------
    W = 75
    print()
    print("=" * W)
    print(f"MODEL COMPARISON REPORT: {asset}")
    print("=" * W)
    print(f"\nBacktest: {test_size} days, walk-forward expanding window\n")

    hdr = (
        f"{'Model':<42} {'MSE':>9} {'Dir Acc':>9} "
        f"{'CI Cov':>9} {'VaR Breach':>11}"
    )
    print(hdr)
    print("-" * W)

    def _fmt(v, pct=False):
        if np.isnan(v):
            return "    N/A"
        return f"{v:>8.1%}" if pct else f"{v:>9.5f}"

    for name, m in all_metrics.items():
        line = (
            f"{name:<42} "
            f"{_fmt(m['MSE'])} "
            f"{_fmt(m['Dir Acc'], pct=True)} "
            f"{_fmt(m['CI Coverage (95%)'], pct=True)} "
            f"{_fmt(m['VaR Breach (95%)'], pct=True)}"
        )
        print(line)

    print("-" * W)
    print()
    print("Metric Interpretation:")
    print("  MSE        : lower = better point-forecast accuracy")
    print("  Dir Acc    : % correct sign prediction  (50% = random, 52-55% = meaningful edge)")
    print("  CI Cov     : % actuals inside 95% CI    (target ≈ 95%)")
    print("  VaR Breach : % actuals breaching VaR_95 (target ≈ 5%)")
    print()

    # Winners (only among models that have the metric)
    def _winner(key, maximize=False):
        candidates = {k: v[key] for k, v in all_metrics.items() if not np.isnan(v.get(key, float("nan")))}
        if not candidates:
            return "N/A"
        return max(candidates, key=candidates.__getitem__) if maximize else min(candidates, key=candidates.__getitem__)

    def _risk_winner():
        candidates = {
            k: abs(v.get("CI Coverage (95%)", float("nan")) - 0.95)
               + abs(v.get("VaR Breach (95%)", float("nan")) - 0.05)
            for k, v in all_metrics.items()
            if not np.isnan(v.get("CI Coverage (95%)", float("nan")))
        }
        return min(candidates, key=candidates.__getitem__) if candidates else "N/A"

    print("Winners:")
    print(f"  Best Forecast Accuracy (MSE) : {_winner('MSE')}")
    print(f"  Best Direction Prediction    : {_winner('Dir Acc', maximize=True)}")
    print(f"  Best Risk Calibration        : {_risk_winner()}")
    print("=" * W)

    return pd.DataFrame(all_metrics).T
