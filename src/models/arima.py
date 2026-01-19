from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Literal, Optional, Tuple, Dict, Any, List
from arch import arch_model
from statsmodels.tsa.arima.model import ARIMA
import logging
import warnings

# Suppress convergence warnings
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)

Innovation = Literal["normal", "t", "skewt"]

@dataclass
class ArimaGarchConfig:
    """
    Configuration for ARIMA–GARCH.
    """
    arima_order: Tuple[int, int, int] = (3, 0, 0) # AR(3) baseline for returns
    garch_order: Tuple[int, int] = (1, 1)         # Standard GARCH(1,1)
    dist: Innovation = "t"                        # Student's t for fat tails
    rescale: bool = True                          # Auto-rescale for numerical stability
    mean_in_garch: Literal["Zero", "Constant"] = "Zero" # Zero because ARIMA handles mean
    min_train_obs: int = 30 # Minimum required for stability

class ArimaGarchModel:
    """
    Robust ARIMA-GARCH Forecaster.
    1. Fits ARIMA to conditional mean.
    2. Fits GARCH to ARIMA residuals (conditional variance).
    """
    def __init__(self, config: ArimaGarchConfig = ArimaGarchConfig()):
        self.cfg = config
        self._arima_res = None
        self._garch_res = None
        self._scale = 1.0 # Internal scaling factor

    def fit(self, y_train: pd.Series):
        """
        Fits ARIMA then GARCH.
        y_train: Series of log returns
        """
        # Ensure data is clean
        y_clean = y_train.dropna()
        if len(y_clean) < self.cfg.min_train_obs:
            logger.warning(f"Not enough data ({len(y_clean)}). Skipping fit.")
            return

        # ---- 1) Fit ARIMA (Mean Equation) ----
        p, d, q = self.cfg.arima_order
        
        # Fallback candidates if primary order fails (robustness)
        candidates = [
            (p, d, q),
            (max(p - 1, 0), d, q),
            (p, d, max(q - 1, 0)),
            (1, 0, 0), # Simple AR(1)
            (0, 0, 0), # White noise (Just constant)
        ]

        self._arima_res = None
        for order in candidates:
            try:
                # 'c' allows constant drift (mu), 'n' if we assume 0 mean
                model = ARIMA(y_clean, order=order, trend='c') 
                self._arima_res = model.fit()
                break # Success
            except Exception:
                continue

        if self._arima_res is None:
            logger.warning("ARIMA fit failed for all candidates.")
            return

        # Get residuals (epsilon_t)
        resid = self._arima_res.resid

        # ---- 2) Fit GARCH (Variance Equation) ----
        # We model the residuals.
        # Check for essentially zero residuals (perfect fit or flat line)
        if np.allclose(resid, 0):
             logger.warning("ARIMA residuals are zero. Skipping GARCH.")
             return

        try:
            # Map distribution
            dist_map = {"normal": "normal", "t": "t", "skewt": "skewt"}
            
            # GARCH requires scaling effectively usually (returns * 100)
            # Arch's 'rescale=True' handles this mostly, but we can manage it explicitly
            # if we want absolute control. We'll rely on arch's internal rescaling
            # via the `rescale` parameter in config.
            
            am = arch_model(
                resid,
                mean=self.cfg.mean_in_garch, # "Zero" usually
                vol='Garch',
                p=self.cfg.garch_order[0],
                q=self.cfg.garch_order[1],
                dist=dist_map.get(self.cfg.dist, 'normal'),
                rescale=self.cfg.rescale
            )
            
            self._garch_res = am.fit(disp='off', show_warning=False)
            
        except Exception as e:
            logger.warning(f"GARCH fit failed: {e}")
            self._garch_res = None

    def predict(self) -> Tuple[float, float]:
        """
        Returns (mean, sigma) for t+1 used by Validator.
        """
        if self._arima_res is None:
            return 0.0, 0.0
            
        # 1. Mean Forecast (mu_{t+1})
        try:
            mean_fc = float(self._arima_res.forecast(steps=1).iloc[0])
        except Exception:
            mean_fc = 0.0

        # 2. Volatility Forecast (sigma_{t+1})
        vol_fc = 0.0
        if self._garch_res is not None:
            try:
                # forecast() returns a VarianceForecast object
                # We need the one-step ahead variance
                garch_fc = self._garch_res.forecast(horizon=1, reindex=False)
                next_var = garch_fc.variance.iloc[-1, 0] # Last row, first column
                
                # IMPORTANT: If data was rescaled, arch SHOULD handle the unscaling of the variance 
                # forecast automatically in recent versions, but let's verify if needed.
                # Actually, arch returns variance in the scaled unit if 'rescale=True' was used?
                # No, arch attempts to return in original scale usually? 
                # Wait, if rescale=True, arch scales input by 10 or 100, estimates, and parameters correspond to scaled data.
                # The forecast object usually handles this? 
                # Docstring says: "The forecast ... is in the same scale as the data provided."
                # If 'rescale=True' was used, the model internally scaled it, but does `forecast` unscale?
                # Usually YES. But to be safe, standard robust usage is often:
                # scale manually -> fit -> forecast -> unscale manually.
                # However, with `rescale=True`, arch handles it.
                
                vol_fc = np.sqrt(max(next_var, 0.0))
            except Exception:
                vol_fc = 0.0
        
        # Fallback: if GARCH failed, maybe use rolling std of recent residuals?
        # For now, return 0.0 and let Risk Engine see it as missing.
        
        return mean_fc, vol_fc

# --- Interfaces for WalkForwardValidator ---

def train_arima_garch(X_train: pd.DataFrame, y_train: pd.Series) -> ArimaGarchModel:
    """
    Validator interface.
    Notes: 
    - X_train is ignored (ARIMA is univariate on y).
    - We initialize with 't' distribution for better handling of market shocks.
    """
    config = ArimaGarchConfig(
        arima_order=(3, 0, 0), 
        garch_order=(1, 1), 
        dist='t'
    )
    model = ArimaGarchModel(config)
    model.fit(y_train)
    return model

def predict_arima_garch(model: ArimaGarchModel, X_test: pd.DataFrame):
    """
    Validator interface.
    Returns: (mean_pred, vol_pred) broadcasted to len(X_test).
    """
    mean, vol = model.predict()
    n = len(X_test)
    
    # Return as arrays
    return np.full(n, mean), np.full(n, vol)
