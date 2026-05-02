"""
BLS (Bureau of Labor Statistics) direct-API feature pipeline.

Fetches series available only on the BLS public API. These aren't easily
accessible through FRED. Produces stationary, lookahead-safe daily features.

API used
--------
BLS Public Data API v2 (POST)
  URL  : https://api.bls.gov/publicAPI/v2/timeseries/data/
  Auth : optional. Works without a key under reduced limits.

Limits without a key (IP-tracked):
  • 25 requests per day
  • 25 series per request
  • 10-year date range per request (silently truncated if exceeded)

Limits with BLS_API_KEY set:
  • 500 requests per day
  • 50 series per request
  • 20-year date range per request

Usage: set BLS_API_KEY in .env for production. Leave empty for development.
The pipeline detects the key automatically from the environment.

Chunking
--------
For requests spanning > 10 years (or > 20 years with a key), fetch_bls_raw()
splits the range into non-overlapping chunks via macro_utils.fetch_chunked()
and concatenates the results.  This is transparent to the caller.

Release lags
------------
BLS timestamps data by its *reference period* (e.g. January CPI is tagged
2024-01-01).  build_bls_features() shifts every series' index forward by its
known publication lag (calendar days) before forward-filling onto the daily
trading calendar.  This ensures no BLS value appears in the feature matrix
before the date a trader could have observed it.

Series fetched
--------------
  CUSR0000SAH1   Shelter CPI (SA)       → yoy_ShelterCPI   MoM log-diff, lag 15d
  CUSR0000SAM    Medical Care CPI (SA)  → yoy_MedicalCPI   MoM log-diff, lag 15d
  CES0500000003  Avg hourly earnings    → d_HourlyEarnings  MoM log-diff, lag  5d
  JTS00000000JOR Job openings rate      → d_JOLTS           first diff,   lag 35d
"""

import os
import time
import warnings
import requests
import pandas as pd

from macro_utils import (
    _to_date_index,
    apply_release_lag,
    apply_transform,
    fetch_chunked,
    validate_features,
)

# ── Constants ──────────────────────────────────────────────────────────────────

BLS_V2_URL = "https://api.bls.gov/publicAPI/v2/timeseries/data/"

# Key-dependent limits (auto-selected in fetch_bls_raw)
_LIMITS = {
    "no_key":   {"max_series": 25, "max_years": 10},
    "with_key": {"max_series": 50, "max_years": 20},
}

# Series registry
# Each entry: BLS_series_id → {output_col, transform, yoy_lags, release_lag, ffill_limit}
# release_lag: calendar days from reference-period month-start to publication.
# ffill_limit: max trading days to forward-fill on the daily calendar.
BLS_SERIES = {
    "CUSR0000SAH1": {
        "output":       "mom_ShelterCPI",
        "transform":    "log_diff",
        "yoy_lags":     1,
        "release_lag":  15,
        "ffill_limit":  25,
        "description":  "Shelter CPI (SA), MoM log-diff",
    },
    "CUSR0000SAM": {
        "output":       "mom_MedicalCPI",
        "transform":    "log_diff",
        "yoy_lags":     1,
        "release_lag":  15,
        "ffill_limit":  25,
        "description":  "Medical Care CPI (SA), MoM log-diff",
    },
    "CES0500000003": {
        "output":       "d_HourlyEarnings",
        "transform":    "log_diff",
        "yoy_lags":     1,
        "release_lag":  5,
        "ffill_limit":  25,
        "description":  "Avg hourly earnings, private sector, MoM log-diff",
    },
    "JTS00000000JOR": {
        "output":       "d_JOLTS",
        "transform":    "diff",
        "yoy_lags":     1,
        "release_lag":  35,
        "ffill_limit":  25,
        "description":  "Job openings rate (JOLTS), first difference",
    },
}


# ── Private helpers ────────────────────────────────────────────────────────────

def _get_api_key() -> str | None:
    """Return BLS_API_KEY from environment, or None if not set/empty."""
    key = os.environ.get("BLS_API_KEY", "").strip()
    return key if key else None


def _get_limits(api_key: str | None) -> dict:
    """Return the applicable request limits based on whether a key is present."""
    return _LIMITS["with_key"] if api_key else _LIMITS["no_key"]


class BLSRateLimitError(Exception):
    """Raised when the BLS API daily request limit has been reached."""


def _post_bls(
    series_ids: list[str],
    start_year: str,
    end_year: str,
    api_key: str | None = None,
) -> dict:
    """
    Execute a single BLS v2 POST request.

    Parameters
    ----------
    series_ids : list of BLS series IDs (max 25 without key, 50 with key)
    start_year : four-digit year string, e.g. "2015"
    end_year   : four-digit year string, e.g. "2024"
    api_key    : BLS registration key, or None for public/unauthenticated access

    Returns
    -------
    Parsed JSON response dict.  Always checks status == "REQUEST_SUCCEEDED".
    Raises BLSRateLimitError when the daily request quota is exhausted.
    Warns on silent truncations logged in the response's "message" array.
    """
    payload = {
        "seriesid":   series_ids,
        "startyear":  start_year,
        "endyear":    end_year,
    }
    if api_key:
        payload["registrationkey"] = api_key

    headers = {"Content-Type": "application/json"}

    resp = requests.post(BLS_V2_URL, json=payload, headers=headers, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    # Log any informational messages (silent truncations appear here)
    for msg in data.get("message", []):
        warnings.warn(f"BLS API message: {msg}", stacklevel=3)

    status = data.get("status", "")
    if status == "REQUEST_NOT_PROCESSED":
        # Daily rate limit reached. Surface as a typed exception so callers
        # can catch it and degrade gracefully without crashing the pipeline.
        raise BLSRateLimitError(
            f"BLS daily request limit reached (IP quota exhausted). "
            f"Messages: {data.get('message', [])}. "
            f"Register a free key at https://data.bls.gov/registrationEngine/ "
            f"to raise the limit to 500 req/day."
        )
    if status != "REQUEST_SUCCEEDED":
        raise RuntimeError(
            f"BLS API returned status={status!r}. "
            f"Messages: {data.get('message', [])}"
        )

    return data


def _parse_bls_response(resp_json: dict) -> pd.DataFrame:
    """
    Parse the Results.series[].data[] array into a tidy long DataFrame.

    Period codes handled
    --------------------
    M01–M12  : monthly → first day of the month (e.g. M03 2023 → 2023-03-01)
    M13      : annual average. Skipped (not a unique time point).
    Q01–Q04  : quarterly → first day of the quarter
    Others   : skipped with a warning

    Values are strings (may contain commas). Cast to float. Non-numeric
    values (e.g. suppressed "(D)") become NaN.

    Returns
    -------
    DataFrame with columns: [date, series_id, value]
    Indexed 0..N (not by date).  Caller must pivot to wide format.
    """
    records = []
    for series in resp_json["Results"]["series"]:
        sid = series["seriesID"]
        for item in series["data"]:
            year   = int(item["year"])
            period = item["period"]  # e.g. "M01", "M13", "Q01"
            value_str = item["value"].replace(",", "")

            # Parse period → date
            if period.startswith("M"):
                month_num = int(period[1:])
                if month_num == 13:
                    continue          # annual average, skip
                if not (1 <= month_num <= 12):
                    warnings.warn(f"BLS: unexpected period {period!r} for {sid}, skipping")
                    continue
                date = pd.Timestamp(year=year, month=month_num, day=1)
            elif period.startswith("Q"):
                qnum = int(period[1:])
                if not (1 <= qnum <= 4):
                    warnings.warn(f"BLS: unexpected period {period!r} for {sid}, skipping")
                    continue
                month_num = (qnum - 1) * 3 + 1
                date = pd.Timestamp(year=year, month=month_num, day=1)
            else:
                warnings.warn(f"BLS: unrecognised period code {period!r} for {sid}, skipping")
                continue

            # Parse value
            try:
                value = float(value_str)
            except ValueError:
                value = float("nan")

            records.append({"date": date, "series_id": sid, "value": value})

    if not records:
        return pd.DataFrame(columns=["date", "series_id", "value"])

    return pd.DataFrame(records)


def _fetch_bls_chunk(
    series_ids: list[str],
    start_year: str,
    end_year: str,
    api_key: str | None,
    throttle_seconds: float = 1.0,
) -> pd.DataFrame:
    """
    Fetch one time-window for up to max_series IDs, batching as needed.

    If len(series_ids) > max_series_per_request, splits into sub-batches.
    Returns wide DataFrame indexed by date with one column per series_id.
    """
    limits    = _get_limits(api_key)
    max_s     = limits["max_series"]
    all_long  = []

    for i in range(0, len(series_ids), max_s):
        batch = series_ids[i : i + max_s]
        resp  = _post_bls(batch, start_year, end_year, api_key)
        long  = _parse_bls_response(resp)
        all_long.append(long)
        if i + max_s < len(series_ids):
            time.sleep(throttle_seconds)   # avoid hammering the API

    if not all_long:
        return pd.DataFrame()

    combined = pd.concat(all_long, ignore_index=True)
    if combined.empty:
        return pd.DataFrame()

    # Pivot to wide: rows = date, cols = series_id
    wide = combined.pivot_table(
        index="date", columns="series_id", values="value", aggfunc="last"
    )
    wide.index = _to_date_index(wide.index)
    wide.columns.name = None
    return wide


# ── Public API ─────────────────────────────────────────────────────────────────

def fetch_bls_raw(
    start: str,
    end: str,
    api_key: str | None = None,
    extra_buf_months: int = 15,
    cache_dir: str | None = None,
) -> pd.DataFrame:
    """
    Download raw BLS monthly levels for all BLS_SERIES.

    Automatically uses BLS_API_KEY from the environment if not provided.
    Adds a buffer of extra_buf_months before `start` to ensure that
    MoM log-diffs (which consume 1 month) are available from the first
    modelling date.

    Caching
    -------
    If cache_dir is provided (or a .cache/ directory exists next to src/),
    raw data is written to a parquet file on first fetch and reloaded on
    subsequent calls.  This prevents exhausting the BLS daily request limit
    (25/day without a key, 500/day with a key) during development.
    Cache filename encodes the series IDs + date range + api-key-presence
    so different configurations don't collide.

    Chunking
    --------
    Splits the date range into max_years windows to respect BLS API limits
    (10 years without key, 20 years with key).

    Parameters
    ----------
    start            : ISO date string for the modelling start (e.g. "2012-01-01")
    end              : ISO date string for the modelling end   (e.g. "2025-12-31")
    api_key          : BLS registration key, or None (auto-detected from env)
    extra_buf_months : months to fetch before start for transform warmup
    cache_dir        : directory for parquet cache files. None means auto-detect.

    Returns
    -------
    Wide DataFrame indexed by tz-naive midnight month-start dates.
    Columns are BLS series IDs (not yet renamed to output columns).
    NaN where data is unavailable for a given period.
    """
    from pathlib import Path

    if api_key is None:
        api_key = _get_api_key()

    buf_start  = pd.Timestamp(start) - pd.DateOffset(months=extra_buf_months)
    limits     = _get_limits(api_key)
    max_years  = limits["max_years"]
    series_ids = list(BLS_SERIES.keys())

    # ── Cache resolution ──────────────────────────────────────────────────────
    if cache_dir is None:
        # Auto-detect: look for a .cache/ directory relative to this file or cwd
        candidates = [
            Path(__file__).parent.parent / ".cache",
            Path.cwd() / ".cache",
            Path.cwd().parent / ".cache",
        ]
        for c in candidates:
            if c.exists():
                cache_dir = str(c)
                break

    cache_path = None
    if cache_dir:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        key_tag   = "keyed" if api_key else "pub"
        sid_hash  = abs(hash(tuple(sorted(series_ids)))) % (10 ** 8)
        fname     = f"bls_raw_{start[:7]}_{end[:7]}_{key_tag}_{sid_hash}.parquet"
        cache_path = Path(cache_dir) / fname

    if cache_path and cache_path.exists():
        print(f"  [BLS cache] Loading from {cache_path.name}")
        return pd.read_parquet(cache_path)

    # ── Fetch from API ────────────────────────────────────────────────────────
    def _fetch_window(sy: str, ey: str) -> pd.DataFrame:
        return _fetch_bls_chunk(series_ids, sy, ey, api_key)

    try:
        raw = fetch_chunked(_fetch_window, str(buf_start.date()), end, chunk_years=max_years)
    except BLSRateLimitError as e:
        # Daily quota exhausted. Return an empty DataFrame rather than crashing.
        # The pipeline continues. BLS features will be absent from the design matrix.
        warnings.warn(
            f"BLS data unavailable (rate limit): {e}\n"
            "  → BLS features will be skipped this run.  "
            "  Re-run tomorrow or add BLS_API_KEY to .env.",
            stacklevel=2,
        )
        return pd.DataFrame()
    except Exception as e:
        warnings.warn(
            f"BLS fetch failed unexpectedly: {e}\n"
            "  → BLS features will be skipped this run.",
            stacklevel=2,
        )
        return pd.DataFrame()

    # ── Save to cache ─────────────────────────────────────────────────────────
    if cache_path and not raw.empty:
        raw.to_parquet(cache_path)
        print(f"  [BLS cache] Saved to {cache_path.name}")

    return raw


def build_bls_features(
    bls_raw: pd.DataFrame,
    daily_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    Transform raw BLS monthly levels into stationary features aligned to
    the target's daily trading calendar.

    For each series the pipeline is:
      1. Apply stationarity transform at monthly frequency.
      2. Apply publication release lag (shift index forward N calendar days)
         so no value appears before it would be publicly available.
      3. Union-index forward-fill then select trading days:
         a) Build union of shifted source dates and trading dates.
         b) Reindex to union, forward-fill (limit = ffill_limit).
         c) Select only trading-day rows.

    The union-index approach handles the case where post-lag dates fall on
    weekends or holidays (not present in the equity trading calendar).

    Returns
    -------
    pd.DataFrame aligned to daily_index.
    Columns are the output names defined in BLS_SERIES.
    """
    daily_index = _to_date_index(daily_index)
    result_cols = {}

    if bls_raw.empty:
        warnings.warn(
            "build_bls_features received an empty DataFrame. "
            "returning empty features (BLS data unavailable).",
            stacklevel=2,
        )
        return pd.DataFrame(index=daily_index)

    for sid, cfg in BLS_SERIES.items():
        if sid not in bls_raw.columns:
            warnings.warn(
                f"BLS: series {sid!r} absent from raw data, skipping column "
                f"{cfg['output']!r}.",
                stacklevel=2,
            )
            continue

        raw_s = bls_raw[sid].dropna()

        if raw_s.empty:
            warnings.warn(
                f"BLS: series {sid!r} has no non-NaN values, skipping.",
                stacklevel=2,
            )
            continue

        # 1. Stationarity transform at monthly frequency
        transformed = apply_transform(raw_s, cfg["transform"], lags=cfg["yoy_lags"])
        transformed = transformed.dropna()

        if transformed.empty:
            warnings.warn(
                f"BLS: {sid!r} is empty after {cfg['transform']!r} transform, skipping.",
                stacklevel=2,
            )
            continue

        # 2. Apply publication release lag
        transformed_df = transformed.to_frame(name=cfg["output"])
        if cfg["release_lag"] > 0:
            transformed_df = apply_release_lag(transformed_df, cfg["release_lag"])

        # 3. Union-index forward-fill then select trading days
        union_idx = daily_index.union(transformed_df.index).sort_values()
        aligned = (
            transformed_df
            .reindex(union_idx, method=None)
            .ffill(limit=cfg["ffill_limit"])
            .reindex(daily_index)
        )
        result_cols[cfg["output"]] = aligned[cfg["output"]]

    if not result_cols:
        raise RuntimeError(
            "build_bls_features: all BLS series failed. Nothing to return."
        )

    return pd.DataFrame(result_cols, index=daily_index)


def log_bls_config(
    bls_raw: pd.DataFrame,
    bls_features: pd.DataFrame,
) -> None:
    """Print a structured diagnostic log for the BLS block."""
    api_key = _get_api_key()
    tier    = "registered (higher limits)" if api_key else "public / no key"

    print("=" * 65)
    print("BLS DIRECT-API BLOCK CONFIGURATION LOG")
    print("=" * 65)
    print(f"  API tier : {tier}")
    print(f"  Endpoint : {BLS_V2_URL}")

    print(f"\n  {'BLS Series':<18} {'Output Column':<22} {'Transform':<12} "
          f"{'Lag(d)':>6} {'Non-NaN%':>9}")
    print(f"  {'-' * 69}")

    for sid, cfg in BLS_SERIES.items():
        if sid not in bls_raw.columns:
            print(f"  {sid:<18} NOT FETCHED")
            continue
        out_col = cfg["output"]
        if out_col not in bls_features.columns:
            print(f"  {sid:<18} {out_col:<22} absent in features")
            continue
        nonnull_pct = bls_features[out_col].notna().mean() * 100
        print(
            f"  {sid:<18} {out_col:<22} {cfg['transform']:<12} "
            f"{cfg['release_lag']:>6} {nonnull_pct:>8.1f}%"
        )

    print(f"\n  Trading-day rows : {len(bls_features)}")
    print(f"  Feature columns  : {bls_features.shape[1]}")
    print("=" * 65)
