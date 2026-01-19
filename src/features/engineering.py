import pandas as pd
import numpy as np
from typing import Tuple, List, Optional, Union
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)

class TimeSeriesFeatures:
    """
    Generates features for time-series forecasting.
    Includes:
    1. Lagged Returns: r_{t-1} ... r_{t-k}
    2. Rolling Volatility: std(r) over window w
    3. EWMA Volatility
    """
    def __init__(self, lags: int = 5, vol_window: int = 21):
        self.lags = lags
        self.vol_window = vol_window

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Expects a canonical dataframe with 'log_return'.
        Returns a dataframe with features X and target y (next day return).
        Note: This method returns a dataframe with NaNs where lags are not available.
        The caller (or the Scaler wrapper) must handle dropping NaNs.
        """
        out = df.copy()
        
        # 1. Feature: Lagged Returns (r_{t-1} ... r_{t-k})
        # We want to predict r_t using previous information.
        # So X_t contains r_{t-1}, r_{t-2}...
        # In this DF, row 't' currently holds r_t.
        # We will shift the returns to create columns [lag_1, lag_2...]
        for lag in range(1, self.lags + 1):
            out[f'lag_return_{lag}'] = out['log_return'].shift(lag)

        # 2. Feature: Rolling Volatility (Proxy for risk regime)
        # Allows models to condition on recent volatility.
        # Must be strictly backward looking (shift by 1 to exclude current day if we consider t as current)
        # BUT: the 'realized_vol' from Phase 2 (canonical) is already calculated based on past window?
        # Let's re-calculate strict rolling vol here to be explicit about lag.
        # We use purely past returns: window ending at t-1.
        # Shift(1) puts r_{t-1} at current row. Rolling on that excludes r_t.
        
        # Rolling Std Dev
        out['rolling_vol'] = out['log_return'].shift(1).rolling(window=self.vol_window).std()
        
        # EWMA Vol
        # Span = 2/alpha - 1. For a generic "risk" proxy.
        out['ewma_vol'] = out['log_return'].shift(1).ewm(span=self.vol_window).std()

        # 3. Target: Next Period Return
        # We want to predict r_t.
        # The dataframe currently has r_t in 'log_return'.
        # So for row t, features are lags (t-1...t-k) and target is log_return (t).
        # We don't shift the target, we just identify it.
        # Rename log_return to target for clarity in X/y split?
        # Actually standard practice: X are the features, y is 'log_return'.
        
        return out.dropna()

class StrictScaler:
    """
    Wrapper to ensure scaling parameters (mu, sigma) are computed ONLY on training data.
    """
    def __init__(self):
        self.scaler = StandardScaler()
        self.is_fitted = False

    def fit(self, X: pd.DataFrame):
        """Fit scaler on Training set X."""
        self.scaler.fit(X)
        self.is_fitted = True

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply transform to any set (Train, Val, Test)."""
        if not self.is_fitted:
            raise ValueError("Scaler must be fit on training data first.")
        
        scaled = self.scaler.transform(X)
        return pd.DataFrame(scaled, index=X.index, columns=X.columns)

def create_tabular_dataset(df: pd.DataFrame, lags: int=5) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Orchestrator for XGBoost data prep.
    Returns: X (features), y (target)
    """
    ts_gen = TimeSeriesFeatures(lags=lags)
    # Generate lags + vol
    full_df = ts_gen.transform(df)
    
    # Define Feature Columns (exclude target, date, non-numeric)
    feature_cols = [c for c in full_df.columns if c.startswith('lag_') or 'vol' in c]
    
    X = full_df[feature_cols]
    y = full_df['log_return'] 
    
    return X, y

def create_sequence_dataset(
    X_scaled: pd.DataFrame, 
    y: pd.Series, 
    seq_length: int=10
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reshape tabular data into sequences for LSTM.
    Input X_scaled should already be scaled.
    Output X shape: (Samples, Seq_Length, Features)
    Output y shape: (Samples,)
    """
    X_vals = X_scaled.values
    y_vals = y.values
    
    X_seq = []
    y_seq = []
    
    # We need L steps of history to predict target at t.
    # range from seq_length to end
    for i in range(seq_length, len(X_vals)):
        # Sequence: X from t-L to t-1
        # Target: y at t
        X_seq.append(X_vals[i-seq_length:i]) 
        y_seq.append(y_vals[i])
        
    return np.array(X_seq), np.array(y_seq)
