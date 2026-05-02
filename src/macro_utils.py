"""
Shared utilities for all macro data pipelines.

Every pipeline module (fred_pipeline, bls_pipeline, bea_pipeline, news_pipeline)
imports from here to guarantee a consistent index contract, release-lag encoding,
stationarity transforms, and diagnostic validation.

Index contract
--------------
All pipeline outputs have a tz-naive midnight DatetimeIndex.
Enforced by _to_date_index() at both fetch time and alignment time.

Release lag
-----------
Monthly/quarterly data timestamps represent the *reference period* end, NOT the
publication date.  apply_release_lag() shifts the index forward by the known
publication lag so that no value appears in the feature matrix before it would
have been publicly available to a trader.

Stationarity transforms
-----------------------
apply_transform() dispatches by transform name:
  "diff"            : first difference (for yield/rate levels)
  "log_diff"        : log first difference (for price/activity indices)
  "yoy_log_diff"    : year-over-year log difference (for money supply, CPI, GDP)
  "d_yoy_log_diff"  : diff of year-over-year log diff. Captures the change in
                       annual inflation. Stationary under structural breaks
                       like the 2021-2023 inflation spike.
  "pct_change"      : simple percentage change
  "none"            : identity. Series is already stationary (e.g. NFCI, STLFSI4)
"""

import numpy as np
import pandas as pd


# ── Index normalisation ────────────────────────────────────────────────────────

def _to_date_index(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """
    Normalise any DatetimeIndex to tz-naive midnight dates.

    Handles three cases:
      • tz-aware (any tz)        → tz_convert(None) then normalize()
      • tz-naive with time part  → normalize() (floor to 00:00:00)
      • tz-naive midnight        → no-op
    """
    idx = pd.DatetimeIndex(idx)
    if idx.tz is not None:
        idx = idx.tz_convert(None)
    return idx.normalize()


# ── Release lag ────────────────────────────────────────────────────────────────

def apply_release_lag(df: pd.DataFrame, lag_days: int) -> pd.DataFrame:
    """
    Shift index forward by lag_days calendar days.

    Converts reference-period timestamps to first-available timestamps.
    Example: January CPI tagged 2024-01-01 + 15-day lag → available 2024-01-16.
    Only rows at or after their shifted date will be forward-filled onto
    the trading calendar, eliminating look-ahead bias.

    Parameters
    ----------
    df       : DataFrame indexed by reference-period dates (tz-naive midnight)
    lag_days : calendar days from period end to expected publication

    Returns
    -------
    Copy of df with index shifted forward and re-normalised to midnight.
    """
    df = df.copy()
    df.index = _to_date_index(df.index + pd.Timedelta(days=lag_days))
    return df


# ── Stationarity transforms ────────────────────────────────────────────────────

def apply_transform(
    series: pd.Series,
    transform: str,
    lags: int = 1,
) -> pd.Series:
    """
    Apply a named stationarity transform to a Series.

    Parameters
    ----------
    series    : raw level Series (any frequency)
    transform : one of {"diff", "log_diff", "yoy_log_diff", "d_yoy_log_diff", "pct_change", "none"}
    lags      : number of periods for diff-based transforms.
                For yoy_log_diff on monthly data use lags=12.
                On quarterly data use lags=4.
                Ignored for "none" and "pct_change".

    Returns
    -------
    Transformed Series (same index, first `lags` values are NaN).
    """
    if transform == "diff":
        return series.diff(lags)
    elif transform == "log_diff":
        return np.log(series.clip(lower=1e-10)).diff(lags)
    elif transform == "yoy_log_diff":
        return np.log(series.clip(lower=1e-10)).diff(lags)
    elif transform == "d_yoy_log_diff":
        # First difference of the year-over-year log change.
        # For quarterly data (lags=4) this captures the *change* in annual
        # inflation. Stays stationary even during structural breaks like the
        # 2021-2023 inflation spike.
        return np.log(series.clip(lower=1e-10)).diff(lags).diff()
    elif transform == "pct_change":
        return series.pct_change(lags)
    elif transform == "none":
        return series.copy()
    else:
        raise ValueError(
            f"Unknown transform {transform!r}. "
            "Choose from: 'diff', 'log_diff', 'yoy_log_diff', "
            "'d_yoy_log_diff', 'pct_change', 'none'."
        )


# ── Chunked year-range fetch ───────────────────────────────────────────────────

def fetch_chunked(
    fetch_fn,
    start: str,
    end: str,
    chunk_years: int = 10,
) -> pd.DataFrame:
    """
    Split a multi-decade date range into non-overlapping chunks and concatenate.

    Necessary for APIs with a maximum year-range per request (e.g. BLS v2
    without a key: 10-year max).  Deduplicates on the DatetimeIndex and
    returns rows sorted ascending.

    Parameters
    ----------
    fetch_fn    : callable(start_year: str, end_year: str) -> pd.DataFrame
                  Must return a DataFrame with a DatetimeIndex.
    start       : ISO date string for the overall start (e.g. "2010-01-01")
    end         : ISO date string for the overall end   (e.g. "2025-12-31")
    chunk_years : maximum year span per call

    Returns
    -------
    Concatenated, deduplicated DataFrame sorted by index.
    Raises ValueError if fetch_fn returns an empty DataFrame for every chunk.
    """
    start_year = pd.Timestamp(start).year
    end_year   = pd.Timestamp(end).year
    frames     = []
    current    = start_year

    while current <= end_year:
        chunk_end = min(current + chunk_years - 1, end_year)
        chunk_df  = fetch_fn(str(current), str(chunk_end))
        if not chunk_df.empty:
            frames.append(chunk_df)
        current = chunk_end + 1

    if not frames:
        raise ValueError(
            f"fetch_chunked: fetch_fn returned no data for any chunk "
            f"in [{start_year}, {end_year}]."
        )

    result = pd.concat(frames, axis=0)
    result = result[~result.index.duplicated(keep="last")]
    return result.sort_index()


# ── Diagnostic validation ──────────────────────────────────────────────────────

def validate_features(
    df: pd.DataFrame,
    label: str,
    min_nonnull_frac: float = 0.50,
    run_adf: bool = True,
) -> None:
    """
    Print a structured diagnostic report and raise AssertionError on failures.

    Checks performed
    ----------------
    1. Non-NaN fraction per column >= min_nonnull_frac
    2. ADF stationarity (p < 0.05) for all columns with >= 20 non-NaN values
    3. No column is all-zero (would silently pass ADF)

    Parameters
    ----------
    df              : aligned feature DataFrame (tz-naive midnight index)
    label           : human-readable name for this data block (printed in header)
    min_nonnull_frac: minimum acceptable non-NaN fraction (default 0.50)
    run_adf         : whether to run the ADF test (default True)

    Raises
    ------
    AssertionError if any check fails, with a summary of all failures.
    """
    from statsmodels.tsa.stattools import adfuller

    print(f"\n{'=' * 65}")
    print(f"DIAGNOSTIC. {label}")
    print(f"{'=' * 65}")
    print(f"  {'Column':<28} {'Non-NaN%':>9} {'ADF p':>8}  {'Pass?':>6}")
    print(f"  {'-' * 56}")

    failures = []

    for col in df.columns:
        s            = df[col].dropna()
        nonnull_frac = len(s) / max(len(df), 1)

        # Non-NaN check
        if nonnull_frac < min_nonnull_frac:
            failures.append(
                f"{col}: non-NaN = {nonnull_frac:.1%} < threshold {min_nonnull_frac:.0%}"
            )

        # All-zero check
        if nonnull_frac > 0 and (s == 0).all():
            failures.append(f"{col}: all values are zero")

        # ADF stationarity check
        if run_adf and len(s) >= 20:
            try:
                adf_p = adfuller(s, autolag="AIC")[1]
                stationary_str = "✓" if adf_p < 0.05 else "✗ FAIL"
                if adf_p >= 0.05:
                    failures.append(f"{col}: ADF p = {adf_p:.4f} (non-stationary)")
            except ValueError as e:
                err = str(e).lower()
                if "constant" in err or "collinear" in err or "singular" in err:
                    # Constant or near-constant series. Trivially stationary, skip ADF.
                    adf_p = float("nan")
                    stationary_str = "const"
                else:
                    adf_p = float("nan")
                    stationary_str = "ERR"
                    failures.append(f"{col}: ADF raised ValueError: {e}")
        else:
            adf_p          = float("nan")
            stationary_str = "-"

        print(
            f"  {col:<28} {nonnull_frac:>8.1%}  {adf_p:>7.3f}  {stationary_str:>6}"
        )

    print(f"\n  Rows: {len(df)}  |  Columns: {df.shape[1]}")

    if failures:
        msg = "\n".join(f"  • {f}" for f in failures)
        raise AssertionError(
            f"\nDiagnostic FAILED. [{label}], {len(failures)} issue(s):\n{msg}"
        )

    print("  All checks passed ✓")
    print(f"{'=' * 65}")
