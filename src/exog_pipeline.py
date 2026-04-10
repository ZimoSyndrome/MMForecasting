"""
Exogenous variable pipeline for ARIMAX-GARCH models.
Fetches market + sector ETF log returns and constructs lagged feature matrices.
"""

import numpy as np
import pandas as pd
import yfinance as yf
import warnings
warnings.filterwarnings("ignore")

MARKET_TICKER = "SPY"
SECTOR_ETFS = ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY", "XLC"]
ALL_EXOG_CANDIDATES = [MARKET_TICKER] + SECTOR_ETFS


def determine_exog_tickers(target_ticker: str):
    """
    Returns (included, excluded, reasons).
    Excludes the target ticker from the exogenous set (no self-regression).
    Also classifies each ticker for logging.
    """
    target_upper = target_ticker.upper()
    included, excluded, reasons = [], [], {}

    for t in ALL_EXOG_CANDIDATES:
        if t.upper() == target_upper:
            excluded.append(t)
            reasons[t] = "Self-regression exclusion: matches target ticker"
        else:
            included.append(t)

    return included, excluded, reasons


def fetch_exog_returns(tickers: list, start: str, end: str) -> pd.DataFrame:
    """
    Download daily adjusted close prices for multiple tickers.
    Returns a DataFrame of log returns (dates × tickers), NaN rows where
    all tickers are missing are dropped.
    """
    if len(tickers) == 1:
        raw = yf.download(tickers[0], start=start, end=end,
                          progress=False, auto_adjust=True)
        if isinstance(raw.columns, pd.MultiIndex):
            prices = raw["Close"]
        else:
            prices = raw[["Close"]]
            prices.columns = tickers
    else:
        raw = yf.download(tickers, start=start, end=end,
                          progress=False, auto_adjust=True)
        if isinstance(raw.columns, pd.MultiIndex):
            prices = raw["Close"]
        else:
            prices = raw[["Close"]].copy()
            prices.columns = tickers

    # Ensure column names are strings (yfinance can return tuples)
    prices.columns = [str(c) for c in prices.columns]
    prices = prices.dropna(how="all")
    log_returns = np.log(prices / prices.shift(1))
    return log_returns.dropna(how="all")


def build_full_design_matrix(
    target_series: pd.Series,
    exog_returns: pd.DataFrame,
    n_ar_lags: int = 5,
    n_exog_lags: int = 5,
):
    """
    Build the joint aligned design matrix (y, X_ar, X_exog_lags).

    All features use strictly lagged data — no lookahead:
      - y_lag1 .. y_lag{n_ar_lags}  : AR lags of the target return
      - {ticker}_lag1 .. lag{n_exog_lags} : lagged returns for each exog ticker

    Returns
    -------
    y       : pd.Series  — aligned target returns
    X_ar    : pd.DataFrame — AR lag columns
    X_exog  : pd.DataFrame — exog lag columns
    """
    # AR lags of target
    ar_cols = {f"y_lag{i}": target_series.shift(i) for i in range(1, n_ar_lags + 1)}
    ar_df = pd.DataFrame(ar_cols, index=target_series.index)

    # Exogenous lags (per ticker, per lag depth)
    exog_lag_parts = []
    for col in exog_returns.columns:
        for lag in range(1, n_exog_lags + 1):
            s = exog_returns[col].shift(lag)
            s.name = f"{col}_lag{lag}"
            exog_lag_parts.append(s)
    exog_lag_df = pd.concat(exog_lag_parts, axis=1)

    # Align everything on dates, drop any row with a NaN
    combined = pd.concat(
        [target_series.rename("y"), ar_df, exog_lag_df], axis=1
    ).dropna()

    ar_cols_list = list(ar_df.columns)
    exog_cols_list = list(exog_lag_df.columns)

    return combined["y"], combined[ar_cols_list], combined[exog_cols_list]


def log_exog_config(
    target: str,
    included: list,
    excluded: list,
    reasons: dict,
    pca_k: int,
    n_components_pls: int,
    n_ar_lags: int,
    n_exog_lags: int,
    design_shape: tuple,
    effective_start: str,
):
    """Print a structured log of the exogenous variable configuration."""
    if target.upper() == MARKET_TICKER:
        ttype = "market index"
    elif target.upper() in SECTOR_ETFS:
        ttype = "sector ETF"
    else:
        ttype = "individual stock"

    raw_exog_features = len(included) * n_exog_lags

    print("=" * 65)
    print("EXOGENOUS VARIABLE CONFIGURATION")
    print("=" * 65)
    print(f"Target           : {target} ({ttype})")
    print(f"Effective start  : {effective_start}  (after lag/NaN drop)")
    print(f"Design matrix    : {design_shape[0]} obs × "
          f"{design_shape[1] + design_shape[2]} features "
          f"({design_shape[1]} AR + {design_shape[2]} exog)")
    print()
    print(f"Included exogenous variables ({len(included)}):")
    for t in included:
        src = "market" if t == MARKET_TICKER else "sector ETF"
        print(f"  + {t:<8}  [{src}]")
    if excluded:
        print(f"\nExcluded variables ({len(excluded)}):")
        for t in excluded:
            print(f"  - {t:<8}  Reason: {reasons[t]}")
    print()
    print("Feature construction:")
    print(f"  AR lags per target   : {n_ar_lags}   (y_lag1 .. y_lag{n_ar_lags})")
    print(f"  Exog lags per ticker : {n_exog_lags}   (lag1 .. lag{n_exog_lags})")
    print(f"  Raw exog features    : {len(included)} × {n_exog_lags} = {raw_exog_features}")
    print()
    print("Dimensionality reduction:")
    print(f"  PCA+EN  — k={pca_k} components, then Elastic Net sparse selection")
    print(f"  PLS     — c={n_components_pls} latent components (supervised)")
    print("=" * 65)
