import pandas as pd
import numpy as np
import logging
from dataclasses import dataclass
from typing import Optional, Literal

logger = logging.getLogger(__name__)

@dataclass
class BacktestConfig:
    initial_capital: float = 10000.0
    cost_bps: float = 0.0010  # 10 bps
    signal_logic: Literal["long_only", "long_short"] = "long_only"
    vol_target: Optional[float] = None  # Annualized vol target (e.g. 0.15)
    max_leverage: float = 1.0

class BacktestEngine:
    """
    Deterministic Backtester.
    Input: DataFrame with ['date', 'actual', 'predicted', 'pred_vol']
    Output: DataFrame with ['signal', 'position', 'return', 'pnl', 'equity']
    """
    def __init__(self, config: BacktestConfig = BacktestConfig()):
        self.cfg = config

    def run(self, results: pd.DataFrame) -> pd.DataFrame:
        """
        Execute backtest on Walk-Forward Results.
        """
        df = results.sort_values('date').copy()
        
        # 1. Generate Signal (s_t) derived from Forecast at t-1 (predicted column)
        # Note: 'predicted' for row T is the forecast MADE at T-1 for T.
        # So we can use logic: if predicted > 0, we want to be Long at T.
        # Wait, let's verify alignment.
        # Validator stores: date=T, actual=r_T, predicted=f(T-1) -> \hat{r}_T.
        # So at start of day T, we see \hat{r}_T. We execute trade at Open T.
        # The return we capture is r_T.
        # So Position_T depends on Predicted_T.
        
        # Logic: 
        # If predicted > 0 -> Signal = 1
        # Else -> Signal = 0 (for Long Only)
        
        if self.cfg.signal_logic == "long_only":
            df['signal'] = np.where(df['predicted'] > 0, 1.0, 0.0)
        else:
            # Long-Short
            df['signal'] = np.sign(df['predicted']) # 1, -1, 0
            
        # 2. Position Sizing
        # Baseline: w_t = s_t
        df['weight'] = df['signal']
        
        # Optional: Vol Targeting
        # w_t = s_t * (Target / ForecastVol)
        if self.cfg.vol_target is not None and 'pred_vol' in df.columns:
            # Handle missing or zero vol
            vol_safe = df['pred_vol'].replace(0, np.nan).ffill()
            # Annualize scalar if needed? Assuming pred_vol is daily sigma
            # Target is usually annualized (e.g. 0.15).
            # Daily Sigma ~ Ann / 16. So TargetDaily = TargetAnn / 16.
            # Or assume inputs are matched. Let's assume params are annualized 
            # and pred_vol is daily.
            target_daily = self.cfg.vol_target / np.sqrt(252)
            
            # leverage factor
            lev = target_daily / vol_safe
            df['weight'] = df['weight'] * lev
            
            # Cap leverage
            df['weight'] = df['weight'].clip(-self.cfg.max_leverage, self.cfg.max_leverage)
            
        # Fill NaN weights (e.g. start) with 0
        df['weight'] = df['weight'].fillna(0.0)

        # 3. PnL Calculation
        # Return_T = Weight_T * Actual_T
        # Note: Weight_T is determined by signal from (T-1 info).
        # In our table, 'predicted' is aligned to 'date' T. 
        # It represents the forecast FOR T.
        # So we can trade on it at T open (or close T-1).
        # Thus: Strategy Return T = Weight T * Actual T
        
        df['strat_ret_gross'] = df['weight'] * df['actual']
        
        # 4. Transaction Costs
        # Cost = bps * delta_weight
        # We need previous weight to calculate turnover
        # Shift weight by 1 to get w_{t-1} relative to row ordering?
        # Typically turnover is |w_t - w_{t-1}|.
        # Since rows are time steps t:
        df['prev_weight'] = df['weight'].shift(1).fillna(0.0)
        df['turnover'] = (df['weight'] - df['prev_weight']).abs()
        
        df['cost'] = df['turnover'] * self.cfg.cost_bps
        
        df['strat_ret_net'] = df['strat_ret_gross'] - df['cost']
        
        # 5. Equity Curve
        # R_t_net
        df['equity_curve'] = (1 + df['strat_ret_net']).cumprod() * self.cfg.initial_capital
        
        # Cleanup
        out_cols = ['date', 'actual', 'predicted', 'signal', 'weight', 'turnover', 'cost', 'strat_ret_net', 'equity_curve']
        if 'pred_vol' in df.columns:
            out_cols.insert(3, 'pred_vol')
            
        return df[out_cols]
