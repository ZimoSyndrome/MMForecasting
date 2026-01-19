import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
from dataclasses import dataclass

@dataclass
class RiskMetrics:
    annualized_return: float
    annualized_vol: float
    sharpe_ratio: float
    max_drawdown: float
    var_95: float
    cvar_95: float
    calmar_ratio: float
    sortino_ratio: float

class RiskEngine:
    """
    Computes rigorous risk metrics on realized returns.
    """
    def __init__(self, risk_free_rate: float = 0.0, periods_per_year: int = 252):
        self.rf = risk_free_rate
        self.periods = periods_per_year

    def calculate(self, returns: pd.Series) -> Dict[str, Any]:
        """
        Compute full suite of scalar risk metrics.
        Input: Realized returns (typically 'strat_ret_net').
        """
        # Ensure clean clean data
        r = returns.dropna()
        if len(r) < 2:
            return {}

        # 1. Basic Stats
        mean_ret = r.mean()
        vol = r.std(ddof=1)
        
        # 2. Annualization
        ann_ret = (1 + mean_ret) ** self.periods - 1
        ann_vol = vol * np.sqrt(self.periods)
        
        # 3. Sharpe Ratio
        if ann_vol > 0:
            sharpe = (ann_ret - self.rf) / ann_vol
        else:
            sharpe = 0.0
            
        # 4. Drawdown Analysis
        cum_ret = (1 + r).cumprod()
        peak = cum_ret.cummax()
        dd = (cum_ret - peak) / peak
        max_dd = dd.min() # Negative number
        
        # 5. Tail Risk (VaR / CVaR)
        # 95% Historical VaR
        # Percentile 5%
        var_95 = np.percentile(r, 5) # Negative number representing the loss
        
        # CVaR (Average of returns <= VaR)
        cvar_95 = r[r <= var_95].mean()
        
        # 6. Sortino Ratio
        # Downside deviation
        downside_r = r[r < 0]
        downside_std = np.sqrt((downside_r**2).mean()) * np.sqrt(self.periods)
        sortino = (ann_ret - self.rf) / downside_std if downside_std > 0 else 0.0

        # 7. Calmar Ratio
        calmar = ann_ret / abs(max_dd) if max_dd != 0 else 0.0

        return {
            "annualized_return": ann_ret,
            "annualized_vol": ann_vol,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_dd,
            "var_95": var_95,
            "cvar_95": cvar_95,
            "sortino_ratio": sortino,
            "calmar_ratio": calmar
        }

    def rolling_metrics(self, returns: pd.Series, window: int = 63) -> pd.DataFrame:
        """
        Compute rolling volatility, Sharpe, and Drawdown.
        Window: 63 days ~ 3 months.
        """
        r = returns.dropna()
        
        # Rolling Vol (Annualized)
        roll_vol = r.rolling(window).std() * np.sqrt(self.periods)
        
        # Rolling Mean (Annualized) - Simple approximation
        roll_mean = r.rolling(window).mean()
        roll_ann_ret = (1 + roll_mean)**self.periods - 1
        
        # Rolling Sharpe
        roll_sharpe = (roll_ann_ret - self.rf) / roll_vol
        
        # Rolling Max Drawdown? 
        # Usually Rolling MaxDD is "MaxDD occurring within the last N days"
        # OR "Drawdown status at time t".
        # Let's provide "Current Drawdown" series instead, which is more useful time-series.
        cum_ret = (1 + r).cumprod()
        peak = cum_ret.cummax()
        dd = (cum_ret - peak) / peak
        
        return pd.DataFrame({
            "rolling_vol": roll_vol,
            "rolling_sharpe": roll_sharpe,
            "drawdown": dd
        })
