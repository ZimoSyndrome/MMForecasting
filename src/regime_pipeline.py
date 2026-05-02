"""
Rule-based composite regime pipeline.

Builds a 3-state financial-stress regime variable from three FRED series.
Acts as an alternative compression of macro information. Instead of the
21-channel macro matrix, downstream models see a single continuous
`composite_stress` score plus a discrete `regime_label`.

Series
------
| FRED code | Frequency | Lag | Reference                              |
|-----------|-----------|-----|----------------------------------------|
| T10Y3M    | Daily     | 0d  | Ang & Bekaert (2002), Estrella (1998)  |
| VIXCLS    | Daily     | 0d  | Whaley (2000), Bollerslev et al (2009) |
| NFCI      | Weekly    | 2d  | Brave & Butters (2011)                 |

NFCI is already fetched by fred_pipeline.fetch_fred_extended (Tier B weekly
series with transform="none", since it is already a stationary stress index).
The caller passes it in so we don't re-hit the FRED API.

Composite encoding
------------------
1.  Each series is converted to an expanding percentile rank in [0, 1]
    via pandas.expanding().rank(pct=True). No lookahead possible.
2.  T10Y3M is inverted (higher spread means steeper curve means lower stress).
3.  composite_stress = (VIX_rank + NFCI_rank + (1 - T10Y3M_rank)) / 3.
4.  regime_label is 0 (low stress), 1 (neutral), or 2 (high stress) from
    tercile cuts of composite_stress.

Outputs
-------
build_regime_features returns a DataFrame aligned to `daily_index` with:
  composite_stress : continuous [0, 1]
  regime_label     : discrete {0, 1, 2}

Caller must `shift(1)` before concatenating into any feature matrix. The
stress score known at end-of-day t should only be used to predict the
return on day t+1.
"""

import warnings

import numpy as np
import pandas as pd
import pandas_datareader.data as web

from macro_utils import _to_date_index, apply_release_lag


REGIME_SERIES = [
    {"code": "T10Y3M", "output": "T10Y3M", "release_lag": 0, "ffill_limit": 5},
    {"code": "VIXCLS", "output": "VIXCLS", "release_lag": 0, "ffill_limit": 5},
]


def fetch_regime_raw(start: str, end: str) -> pd.DataFrame:
    """Fetch T10Y3M and VIXCLS from FRED via pandas-datareader.

    Returns a DataFrame with tz-naive midnight index, one column per
    series. Returns an empty DataFrame on total failure.
    """
    frames: dict[str, pd.Series] = {}
    for cfg in REGIME_SERIES:
        code = cfg["code"]
        try:
            s = web.DataReader(code, "fred", start, end)[code]
            s.index = _to_date_index(s.index)
            frames[cfg["output"]] = s.dropna()
        except Exception as e:
            warnings.warn(f"REGIME: fetch {code} failed: {e}")
    if not frames:
        return pd.DataFrame()
    return pd.DataFrame(frames).sort_index()


def build_regime_features(
    regime_raw: pd.DataFrame,
    nfci_series: pd.Series,
    daily_index: pd.DatetimeIndex,
    min_rank_periods: int = 30,
) -> pd.DataFrame:
    """Build composite stress score and 3-state regime label.

    Parameters
    ----------
    regime_raw        : output of fetch_regime_raw (T10Y3M + VIXCLS)
    nfci_series       : NFCI level series from fred_ext_features['NFCI'].
                        Already release-lag-corrected and aligned to
                        daily_index by fred_pipeline.
    daily_index       : full trading calendar (matches df.index in the notebook)
    min_rank_periods  : minimum observations before expanding rank is defined

    Returns
    -------
    DataFrame with columns ['composite_stress', 'regime_label'] on daily_index.
    Early rows (< min_rank_periods) carry NaN and should be filtered by the
    caller's downstream dropna / fillna logic.
    """
    df = pd.DataFrame(index=daily_index)

    # ── Release-lag + ffill-align T10Y3M and VIXCLS ─────────────────────────
    for cfg in REGIME_SERIES:
        col = cfg["output"]
        if regime_raw.empty or col not in regime_raw.columns:
            df[col] = np.nan
            continue
        s = regime_raw[[col]].copy()
        s = apply_release_lag(s, cfg["release_lag"])
        union_idx = daily_index.union(s.index).sort_values()
        df[col] = (s.reindex(union_idx, method=None)
                    .ffill(limit=cfg["ffill_limit"])
                    .reindex(daily_index)[col])

    # ── NFCI: caller provides a daily-aligned series ────────────────────────
    if nfci_series is None or nfci_series.empty:
        df["NFCI"] = np.nan
    else:
        df["NFCI"] = nfci_series.reindex(daily_index)

    # ── Expanding percentile ranks (strictly no lookahead) ──────────────────
    df["T10Y3M_rank"] = df["T10Y3M"].expanding(min_periods=min_rank_periods).rank(pct=True)
    df["VIXCLS_rank"] = df["VIXCLS"].expanding(min_periods=min_rank_periods).rank(pct=True)
    df["NFCI_rank"]   = df["NFCI"].expanding(min_periods=min_rank_periods).rank(pct=True)

    # ── Composite: invert T10Y3M (steep curve = low stress) ─────────────────
    df["composite_stress"] = (
        df["VIXCLS_rank"]
        + df["NFCI_rank"]
        + (1.0 - df["T10Y3M_rank"])
    ) / 3.0

    # ── 3-state label: 0=low, 1=neutral, 2=high stress ──────────────────────
    df["regime_label"] = pd.cut(
        df["composite_stress"],
        bins=[0.0, 1.0 / 3, 2.0 / 3, 1.0 + 1e-9],
        labels=[0, 1, 2],
        right=True,
    ).astype(float)

    return df[["composite_stress", "regime_label"]]
