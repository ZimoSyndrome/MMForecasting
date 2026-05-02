"""
FRED macro-financial feature pipeline.

Provides two independent feature blocks:

CORE block (5 columns), used in the ARIMAX design matrix
---------------------------------------------------------------------
  DGS10   → d_DGS10        Δ10Y Treasury yield
  DGS2    → d_DGS2         Δ2Y Treasury yield
  derived → d_term_spread  Δterm spread (10Y minus 2Y)
  VIXCLS  → d_VIXCLS       ΔVIX
  BAA10Y  → d_BAA10Y       Δcredit spread (BAA minus 10Y)

EXTENDED block (13 columns)
-----------------------------------------------------------------
  Tier A. Daily (release_lag = 0, ffill = 5 trading days):
    T5YIE   → d_T5YIE        Δ5-yr breakeven inflation (TIPS)
    T10YIE  → d_T10YIE       Δ10-yr breakeven inflation (TIPS)
    T5YIFR  → d_T5YIFR       Δ5yr/5yr forward inflation expectation
    DFF     → d_DFF           ΔEffective Fed Funds Rate (daily)

  Tier B. Weekly (release_lag = 2, ffill = 7 trading days):
    STLFSI4 → STLFSI4         St. Louis Financial Stress Index (level)
    NFCI    → NFCI            Chicago Fed Natl Financial Conditions (level)
    WALCL   → d_WALCL         ΔFed balance sheet total assets (log)

  Tier C. Monthly (release_lag per series, ffill = 25 trading days):
    CPIAUCSL  → yoy_CPI       YoY log-diff CPI all-items SA
    CPILFESL  → yoy_CoreCPI   YoY log-diff core CPI (ex food+energy)
    PAYEMS    → d_PAYEMS      MoM log-diff nonfarm payrolls
    UNRATE    → d_UNRATE      Δunemployment rate
    INDPRO    → d_INDPRO      MoM log-diff industrial production
    UMCSENT   → d_UMCSENT     ΔU.Michigan consumer sentiment

All feature values enter the model only through lagged copies (lag ≥ 1),
and monthly series are shifted by their known publication lags so that
no value appears in the feature matrix before it is publicly available.
"""

import pandas as pd
import pandas_datareader.data as web

from macro_utils import (
    _to_date_index,
    apply_release_lag,
    apply_transform,
    validate_features,
)

# ── Core series registry ───────────────────────────────────────────────────────

FRED_RAW_SERIES = {
    "DGS10":  "10Y Treasury Yield (%)",
    "DGS2":   "2Y Treasury Yield (%)",
    "VIXCLS": "VIX (index)",
    "BAA10Y": "Credit Spread BAA−10Y (%)",
}

FRED_FEATURE_NAMES = {
    "d_DGS10":       "Δ10Y yield",
    "d_DGS2":        "Δ2Y yield",
    "d_term_spread": "Δterm spread (10Y−2Y)",
    "d_VIXCLS":      "ΔVIX",
    "d_BAA10Y":      "Δcredit spread (BAA−10Y)",
}

# ── Extended series registry ───────────────────────────────────────────────────
# Each entry:  FRED_code → {output_col, transform, release_lag_days, ffill_limit}
# release_lag: calendar days from reference-period end to expected publication.
# ffill_limit: max forward-fill days on the daily trading calendar.
# yoy lags:    12 for monthly, 1 for daily/weekly.

FRED_EXTENDED_SERIES = {
    # Tier A. Daily ─────────────────────────────────────────────────────
    "T5YIE": {
        "output":       "d_T5YIE",
        "transform":    "diff",
        "yoy_lags":     1,
        "release_lag":  0,
        "ffill_limit":  5,
        "description":  "Δ5-yr breakeven inflation expectation (TIPS)",
    },
    "T10YIE": {
        "output":       "d_T10YIE",
        "transform":    "diff",
        "yoy_lags":     1,
        "release_lag":  0,
        "ffill_limit":  5,
        "description":  "Δ10-yr breakeven inflation expectation (TIPS)",
    },
    "T5YIFR": {
        "output":       "d_T5YIFR",
        "transform":    "diff",
        "yoy_lags":     1,
        "release_lag":  0,
        "ffill_limit":  5,
        "description":  "Δ5yr/5yr forward inflation expectation",
    },
    "DFF": {
        "output":       "d_DFF",
        "transform":    "diff",
        "yoy_lags":     1,
        "release_lag":  0,
        "ffill_limit":  5,
        "description":  "ΔEffective Fed Funds Rate (daily)",
    },
    # Tier B. Weekly ────────────────────────────────────────────────────
    # FRED dates these series on the reference Friday and publishes the
    # Thursday (STLFSI4, +6 days) and Wednesday (NFCI, +5 days).
    "STLFSI4": {
        "output":       "STLFSI4",
        "transform":    "none",
        "yoy_lags":     1,
        "release_lag":  6,
        "ffill_limit":  7,
        "description":  "St. Louis Financial Stress Index (level)",
    },
    "NFCI": {
        "output":       "NFCI",
        "transform":    "none",
        "yoy_lags":     1,
        "release_lag":  5,
        "ffill_limit":  7,
        "description":  "Chicago Fed National Financial Conditions Index (level)",
    },
    "WALCL": {
        "output":       "d_WALCL",
        "transform":    "log_diff",
        "yoy_lags":     1,
        "release_lag":  2,
        "ffill_limit":  7,
        "description":  "ΔFed balance sheet total assets (weekly log-diff)",
    },
    # Tier C. Monthly ───────────────────────────────────────────────────
    "CPIAUCSL": {
        "output":       "mom_CPI",
        "transform":    "log_diff",
        "yoy_lags":     1,
        "release_lag":  15,
        "ffill_limit":  25,
        "description":  "CPI all-items SA, MoM log-diff (stationary)",
    },
    "CPILFESL": {
        "output":       "mom_CoreCPI",
        "transform":    "log_diff",
        "yoy_lags":     1,
        "release_lag":  15,
        "ffill_limit":  25,
        "description":  "Core CPI (ex food+energy), MoM log-diff (stationary)",
    },
    "PAYEMS": {
        "output":       "d_PAYEMS",
        "transform":    "log_diff",
        "yoy_lags":     1,
        "release_lag":  5,
        "ffill_limit":  25,
        "description":  "Nonfarm payrolls, MoM log-diff",
    },
    "UNRATE": {
        "output":       "d_UNRATE",
        "transform":    "diff",
        "yoy_lags":     1,
        "release_lag":  5,
        "ffill_limit":  25,
        "description":  "Unemployment rate, first difference",
    },
    "INDPRO": {
        "output":       "d_INDPRO",
        "transform":    "log_diff",
        "yoy_lags":     1,
        "release_lag":  16,
        "ffill_limit":  25,
        "description":  "Industrial production, MoM log-diff",
    },
    "UMCSENT": {
        "output":       "d_UMCSENT",
        "transform":    "diff",
        "yoy_lags":     1,
        "release_lag":  14,
        "ffill_limit":  25,
        "description":  "U.Michigan consumer sentiment, first difference",
    },
}


# ── Core pipeline ──────────────────────────────────────────────────────────────

def fetch_fred_raw(start: str, end: str) -> pd.DataFrame:
    """
    Download raw FRED levels for the core 4 series.

    Adds a 40-business-day buffer before `start` so first-differences are
    available from the very first modelling day.  Returns a DataFrame indexed
    by tz-naive midnight dates with NaN on non-update days.
    """
    buf_start = pd.Timestamp(start) - pd.offsets.BDay(40)
    frames = {}
    for code in FRED_RAW_SERIES:
        s = web.DataReader(code, "fred", buf_start, end)[code]
        s.index = _to_date_index(pd.DatetimeIndex(s.index))
        frames[code] = s
    return pd.DataFrame(frames)


def build_fred_features(
    fred_raw: pd.DataFrame,
    daily_index: pd.DatetimeIndex,
    max_ffill: int = 5,
) -> pd.DataFrame:
    """
    Transform raw FRED core levels into stationary features aligned to
    the target's daily trading calendar.

    Steps
    -----
    1. Normalise both indexes to tz-naive midnight.
    2. Compute term spread (DGS10 − DGS2).
    3. First-difference all 5 series.
    4. Reindex to daily_index, forward-fill gaps up to max_ffill days.

    Returns
    -------
    pd.DataFrame aligned to daily_index, columns = FRED_FEATURE_NAMES keys.
    """
    df = fred_raw.copy()
    df.index = _to_date_index(df.index)
    daily_index = _to_date_index(daily_index)

    df["term_spread"] = df["DGS10"] - df["DGS2"]

    changes = df.diff().rename(columns={
        "DGS10":       "d_DGS10",
        "DGS2":        "d_DGS2",
        "term_spread": "d_term_spread",
        "VIXCLS":      "d_VIXCLS",
        "BAA10Y":      "d_BAA10Y",
    })[list(FRED_FEATURE_NAMES.keys())]

    return changes.reindex(daily_index, method=None).ffill(limit=max_ffill)


def log_fred_config(
    fred_raw: pd.DataFrame,
    fred_features: pd.DataFrame,
) -> None:
    """Print a structured log for the core FRED block."""
    print("=" * 65)
    print("FRED CORE BLOCK CONFIGURATION LOG")
    print("=" * 65)

    print("\nRaw series downloaded from FRED:")
    for code, desc in FRED_RAW_SERIES.items():
        col = fred_raw[code].dropna()
        print(
            f"  {code:<10} {desc:<35}  "
            f"{col.index[0].date()} → {col.index[-1].date()}  "
            f"({len(col)} obs)"
        )
    print(f"  {'term_spread':<10} Computed: DGS10 − DGS2")

    print("\nFeatures (first differences):")
    print(f"  {'Feature':<20} {'Description':<35} {'Non-NaN':>7}")
    for code, desc in FRED_FEATURE_NAMES.items():
        n = fred_features[code].notna().sum()
        print(f"  {code:<20} {desc:<35} {n:>7}")

    n_missing = int(fred_features.isna().any(axis=1).sum())
    print(f"\nTrading-day rows aligned : {len(fred_features)}")
    print(f"Rows with any NaN        : {n_missing}")
    print("=" * 65)


# ── Extended pipeline ──────────────────────────────────────────────────────────

def fetch_fred_extended(start: str, end: str) -> pd.DataFrame:
    """
    Download raw FRED levels for all extended series.

    Uses a 400-calendar-day buffer before `start` to ensure that:
    - YoY transforms (12-month or 4-quarter lag) have enough history.
    - First-differences are available from the first modelling date.

    Returns a DataFrame indexed by tz-naive midnight dates.  Each column
    is named by its FRED code.  NaN on non-update days (weekends, holidays,
    and months with no observation for weekly/monthly series).
    """
    buf_start = pd.Timestamp(start) - pd.DateOffset(days=400)
    frames = {}
    for code in FRED_EXTENDED_SERIES:
        try:
            s = web.DataReader(code, "fred", buf_start, end)[code]
            s.index = _to_date_index(pd.DatetimeIndex(s.index))
            frames[code] = s
        except Exception as exc:
            # Log and skip rather than crashing the whole fetch
            print(f"  WARNING: could not fetch {code} from FRED. {exc}")
    return pd.DataFrame(frames)


def build_fred_extended_features(
    fred_ext_raw: pd.DataFrame,
    daily_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    Transform raw extended FRED levels into stationary features aligned to
    the target's daily trading calendar.

    For each series the pipeline is:
      1. Apply the stationarity transform at the series' native frequency.
      2. Apply the publication release lag (shift index forward N days)
         so no value appears before it was publicly available.
      3. Reindex to daily_index and forward-fill up to the series' ffill limit.

    Returns
    -------
    pd.DataFrame aligned to daily_index.
    Columns are the output names defined in FRED_EXTENDED_SERIES.
    Columns for series that could not be fetched are absent (no silent NaN columns).
    """
    daily_index = _to_date_index(daily_index)
    result_cols = {}

    for code, cfg in FRED_EXTENDED_SERIES.items():
        if code not in fred_ext_raw.columns:
            print(f"  WARNING: {code} absent from raw data. Skipping.")
            continue

        raw_s = fred_ext_raw[code].dropna()

        # 1. Stationarity transform at native frequency (monthly stays monthly etc.)
        transformed = apply_transform(raw_s, cfg["transform"], lags=cfg["yoy_lags"])
        transformed = transformed.dropna()

        if transformed.empty:
            print(f"  WARNING: {code} is empty after transform. Skipping.")
            continue

        # 2. Apply publication release lag (shift index forward N calendar days).
        #    After the shift, source dates may land on weekends or holidays
        #    that are absent from the equity trading calendar.
        transformed_df = transformed.to_frame(name=cfg["output"])
        if cfg["release_lag"] > 0:
            transformed_df = apply_release_lag(transformed_df, cfg["release_lag"])

        # 3. Union-index forward-fill then select trading days.
        #    Why union? Reindex with method=None does exact-date matching.
        #    Weekly source dates shifted by N days can land on weekends, which
        #    are never in the equity index, so the exact match produces all-NaN.
        #    Union-index approach:
        #      a) Build a combined sorted index (source dates + trading days).
        #      b) Reindex to the union, inserting NaN for dates with no source data.
        #      c) Forward-fill within the union (sparse → dense).
        #      d) Select only trading days from the filled result.
        union_idx = daily_index.union(transformed_df.index).sort_values()
        aligned = (
            transformed_df
            .reindex(union_idx, method=None)     # exact match on the full union
            .ffill(limit=cfg["ffill_limit"])      # fill forward within the union
            .reindex(daily_index)                 # keep only trading days
        )
        result_cols[cfg["output"]] = aligned[cfg["output"]]

    if not result_cols:
        raise RuntimeError(
            "build_fred_extended_features: all series failed. Nothing to return."
        )

    return pd.DataFrame(result_cols, index=daily_index)


def log_fred_extended_config(
    fred_ext_raw: pd.DataFrame,
    fred_ext_features: pd.DataFrame,
) -> None:
    """Print a structured log for the extended FRED block."""
    print("=" * 65)
    print("FRED EXTENDED BLOCK CONFIGURATION LOG")
    print("=" * 65)

    print(f"\n  {'FRED Code':<12} {'Output Column':<18} {'Transform':<16} "
          f"{'Lag(d)':>6} {'Non-NaN%':>9}")
    print(f"  {'-' * 63}")

    for code, cfg in FRED_EXTENDED_SERIES.items():
        if code not in fred_ext_raw.columns:
            print(f"  {code:<12} {'NOT FETCHED':}")
            continue
        out_col = cfg["output"]
        if out_col not in fred_ext_features.columns:
            print(f"  {code:<12} {out_col:<18} column absent in features")
            continue
        nonnull_pct = fred_ext_features[out_col].notna().mean() * 100
        print(
            f"  {code:<12} {out_col:<18} {cfg['transform']:<16} "
            f"{cfg['release_lag']:>6} {nonnull_pct:>8.1f}%"
        )

    print(f"\n  Trading-day rows : {len(fred_ext_features)}")
    print(f"  Feature columns  : {fred_ext_features.shape[1]}")
    print("=" * 65)
