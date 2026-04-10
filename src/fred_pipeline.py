"""
FRED macro-financial feature pipeline.

Downloads 4 FRED series, derives term spread from levels, computes first
differences (changes) for stationarity, aligns to the daily trading calendar
via forward-fill, and returns a clean feature DataFrame ready to be
concatenated with the ETF exogenous matrix in exog_pipeline.py.

Series fetched
--------------
  DGS10   10-year constant-maturity Treasury yield  (% level → change)
  DGS2    2-year constant-maturity Treasury yield   (% level → change)
  VIXCLS  CBOE VIX index                           (index level → change)
  BAA10Y  Moody's BAA corporate - 10Y Treasury     (% level → change)

Derived
-------
  term_spread = DGS10 - DGS2  (computed before differencing)

Final 5 feature columns (all first-difference / change)
---------------------------------------------------------
  d_DGS10        Δ10Y Treasury yield
  d_DGS2         Δ2Y Treasury yield
  d_term_spread  Δterm spread (10Y−2Y)
  d_VIXCLS       ΔVIX
  d_BAA10Y       Δcredit spread (BAA−10Y)
"""

import pandas as pd
import pandas_datareader.data as web

# ── Constants ──────────────────────────────────────────────────────────────────
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


def fetch_fred_raw(start: str, end: str) -> pd.DataFrame:
    """
    Download raw FRED levels.

    Adds a 40-business-day buffer before `start` so first-differences are
    available from the very first modelling day.  Returns a DataFrame indexed
    by date with NaN on days the series was not updated (weekends, holidays).
    """
    buf_start = pd.Timestamp(start) - pd.offsets.BDay(40)
    frames = {}
    for code in FRED_RAW_SERIES:
        s = web.DataReader(code, "fred", buf_start, end)[code]
        s.index = pd.to_datetime(s.index)
        frames[code] = s
    return pd.DataFrame(frames)


def build_fred_features(
    fred_raw: pd.DataFrame,
    daily_index: pd.DatetimeIndex,
    max_ffill: int = 5,
) -> pd.DataFrame:
    """
    Transform raw FRED levels into stationary change features aligned to
    the target's daily trading calendar.

    Steps
    -----
    1.  Compute term spread (DGS10 − DGS2) from raw levels.
    2.  First-difference all 5 series to produce changes.
    3.  Reindex to *daily_index*, forward-filling gaps up to *max_ffill* days
        to handle weekends and Fed holidays without look-ahead bias.
        (The change on a forward-filled day represents the most recent
        published change — no future information is used.)

    Returns
    -------
    pd.DataFrame aligned to *daily_index*, columns = FRED_FEATURE_NAMES keys.
    """
    df = fred_raw.copy()
    df["term_spread"] = df["DGS10"] - df["DGS2"]

    changes = df.diff().rename(columns={
        "DGS10":       "d_DGS10",
        "DGS2":        "d_DGS2",
        "term_spread": "d_term_spread",
        "VIXCLS":      "d_VIXCLS",
        "BAA10Y":      "d_BAA10Y",
    })[list(FRED_FEATURE_NAMES.keys())]

    aligned = (
        changes
        .reindex(daily_index, method=None)
        .ffill(limit=max_ffill)
    )
    return aligned


def log_fred_config(
    fred_raw: pd.DataFrame,
    fred_features: pd.DataFrame,
) -> None:
    """Print a structured log of what was pulled and what is used in modelling."""
    print("=" * 65)
    print("FRED MACRO BLOCK — CONFIGURATION LOG")
    print("=" * 65)

    print("\nRaw series downloaded from FRED:")
    for code, desc in FRED_RAW_SERIES.items():
        col = fred_raw[code].dropna()
        print(f"  {code:<10} {desc:<35}  "
              f"{col.index[0].date()} → {col.index[-1].date()}  "
              f"({len(col)} obs)")
    print(f"  {'term_spread':<10} Computed: DGS10 − DGS2")

    print("\nFeatures used in modelling (first differences / changes):")
    print(f"  {'Feature':<20} {'Description':<35} {'Non-NaN':>7}")
    for code, desc in FRED_FEATURE_NAMES.items():
        n = fred_features[code].notna().sum()
        print(f"  {code:<20} {desc:<35} {n:>7}")

    n_missing = int(fred_features.isna().any(axis=1).sum())
    print(f"\nTotal trading-day rows aligned : {len(fred_features)}")
    print(f"Rows with any NaN after ffill  : {n_missing}")
    print("=" * 65)
