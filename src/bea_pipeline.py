"""
BEA NIPA quarterly data pipeline.

Fetches GDP growth, PCE Price Index, and Corporate Profits after tax from the
Bureau of Economic Analysis NIPA REST API (free key, no chunking needed.
BEA supports Year=ALL in a single request).

Series fetched
--------------
T10101 / A191RL  → GDP_growth_pct    : QoQ % change (already stationary, transform=none)
T20304 / DPCERG  → d_yoy_PCE_PI     : diff of YoY log-diff (4 qtrs). Captures the
                                       change in annual PCE inflation. Stays stationary
                                       under the 2021-23 inflation spike.
T11200 / A055RC  → yoy_CorpProfits   : Corp Profits after tax. yoy log-diff (4 qtrs).

Release lags
------------
GDP / PCE : 30 calendar days after quarter-end (advance estimate)
Corp Profits: 45 calendar days after quarter-end (third estimate)

Index contract
--------------
All outputs are tz-naive midnight DatetimeIndex. Quarter-start dates before
release-lag shift, trading-calendar dates after alignment.

Graceful degradation
--------------------
If BEA_API_KEY is missing or the API returns no data, fetch_bea_raw() returns
an empty DataFrame and build_bea_features() returns an empty DataFrame indexed
by daily_index. Downstream code (notebook cell 38) guards with:
    if "bea_features" in dir() and not bea_features.empty:
"""

import warnings

import numpy as np
import pandas as pd
import requests

from macro_utils import (
    _to_date_index,
    apply_release_lag,
    apply_transform,
    validate_features,
)


BEA_URL = "https://apps.bea.gov/api/data/"

BEA_SERIES = [
    {
        "table":       "T10101",
        "series_code": "A191RL",
        "output":      "GDP_growth_pct",
        "transform":   "none",
        "lags":        1,
        "release_lag": 30,
        "ffill_limit": 70,
        "description": "Real GDP QoQ % change (advance estimate)",
    },
    {
        "table":       "T20304",
        "series_code": "DPCERG",
        "output":      "d_yoy_PCE_PI",
        "transform":   "d_yoy_log_diff",
        "lags":        4,
        "release_lag": 30,
        "ffill_limit": 70,
        "description": "PCE Price Index. Change in YoY log-diff. Stationary under inflation breaks.",
    },
    {
        "table":       "T11200",
        "series_code": "A055RC",
        "output":      "yoy_CorpProfits",
        "transform":   "yoy_log_diff",
        "lags":        4,
        "release_lag": 45,
        "ffill_limit": 70,
        "description": "Corp Profits after tax. YoY log-diff (4 quarters).",
    },
]


# ── Internal helpers ──────────────────────────────────────────────────────────

def _get_bea_nipa(api_key: str, table_name: str) -> list:
    """
    GET a single BEA NIPA table (all quarters, all years).

    Returns the raw Data list (list of dicts with SeriesCode / TimePeriod /
    DataValue keys).  Returns [] on any error and emits a warning.
    """
    params = {
        "UserID":      api_key,
        "method":      "GetData",
        "datasetname": "NIPA",
        "TableName":   table_name,
        "Frequency":   "Q",
        "Year":        "ALL",
        "ResultFormat": "JSON",
    }
    try:
        resp = requests.get(BEA_URL, params=params, timeout=30).json()
    except Exception as exc:
        warnings.warn(f"BEA: HTTP error fetching {table_name}: {exc}")
        return []

    results = resp.get("BEAAPI", {}).get("Results", None)
    if results is None:
        err = resp.get("BEAAPI", {}).get("Error", {})
        warnings.warn(
            f"BEA: no Results for table {table_name}. "
            f"API error: {err.get('APIErrorDescription', 'unknown')}"
        )
        return []

    return results.get("Data", [])


def _parse_bea_table(data_list: list, series_code: str) -> pd.Series:
    """
    Extract rows matching series_code from a BEA Data list.

    TimePeriod format: "2023Q3" → pd.Period → quarter-start Timestamp.
    DataValue: may contain commas ("22,998.5") → stripped before float cast.
    Suppressed values "(D)" → NaN.

    Returns a Series with tz-naive midnight DatetimeIndex (quarter-start dates),
    or an empty Series if no matching rows are found.
    """
    rows = [r for r in data_list if r.get("SeriesCode") == series_code]
    if not rows:
        return pd.Series(dtype=float, name=series_code)

    records = {}
    for r in rows:
        try:
            ts      = pd.Period(r["TimePeriod"], freq="Q").to_timestamp()  # quarter-start
            val_str = r["DataValue"].replace(",", "").strip()
            records[ts] = float(val_str) if val_str not in ("", "(D)") else np.nan
        except (ValueError, KeyError):
            continue

    s       = pd.Series(records, dtype=float, name=series_code)
    s.index = _to_date_index(s.index)
    return s.sort_index()


# ── Public API ────────────────────────────────────────────────────────────────

def fetch_bea_raw(api_key: str) -> pd.DataFrame:
    """
    Fetch all BEA_SERIES tables and parse each series.

    Parameters
    ----------
    api_key : BEA API key (from .env BEA_API_KEY)

    Returns
    -------
    Wide DataFrame keyed by series_code with quarterly DatetimeIndex
    (quarter-start, tz-naive midnight).  Returns an empty DataFrame if all
    fetches fail or api_key is empty.
    """
    if not api_key:
        warnings.warn(
            "BEA_API_KEY is not set. Skipping BEA fetch. "
            "Set BEA_API_KEY in .env to enable GDP/PCE/CorpProfits features."
        )
        return pd.DataFrame()

    frames = {}
    for cfg in BEA_SERIES:
        data = _get_bea_nipa(api_key, cfg["table"])
        s    = _parse_bea_table(data, cfg["series_code"])
        if s.empty:
            warnings.warn(
                f"BEA: no data parsed for {cfg['series_code']} "
                f"({cfg['description']}) from table {cfg['table']}"
            )
        frames[cfg["series_code"]] = s

    if not any(not s.empty for s in frames.values()):
        warnings.warn("BEA: all series returned empty. fetch_bea_raw returning empty DataFrame.")
        return pd.DataFrame()

    return pd.DataFrame(frames)


def build_bea_features(
    bea_raw: pd.DataFrame,
    daily_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    Apply release lag, stationarity transform, and align to daily trading calendar.

    For each series in BEA_SERIES:
      1. Transform at quarterly frequency (yoy_log_diff over 4 quarters, or none)
      2. apply_release_lag (30–45 calendar days) to prevent lookahead
      3. Union-index forward-fill onto daily_index (limit=70 trading days)

    Parameters
    ----------
    bea_raw    : output of fetch_bea_raw(). Wide quarterly DataFrame.
    daily_index: tz-naive midnight DatetimeIndex of the trading calendar

    Returns
    -------
    Wide DataFrame aligned to daily_index. Returns empty DataFrame (same index)
    if bea_raw is empty.
    """
    if bea_raw.empty:
        return pd.DataFrame(index=daily_index)

    cols = {}
    for cfg in BEA_SERIES:
        sc = cfg["series_code"]
        if sc not in bea_raw.columns:
            continue
        s = bea_raw[sc].dropna()
        if s.empty:
            continue

        # Transform at quarterly frequency before aligning
        transformed    = apply_transform(s, cfg["transform"], lags=cfg["lags"])
        transformed_df = transformed.to_frame(name=cfg["output"])

        # Shift index forward by release lag (reference date → publication date)
        transformed_df = apply_release_lag(transformed_df, cfg["release_lag"])

        # Union-index forward-fill: handles publication dates landing on weekends
        union_idx = daily_index.union(transformed_df.index).sort_values()
        aligned   = (
            transformed_df
            .reindex(union_idx, method=None)
            .ffill(limit=cfg["ffill_limit"])
            .reindex(daily_index)
        )
        cols[cfg["output"]] = aligned[cfg["output"]]

    return pd.DataFrame(cols, index=daily_index)
