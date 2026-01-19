import pandas as pd
import numpy as np
from src.risk.metrics import RiskEngine

def main():
    print("--- Testing Risk Engine ---")
    
    # 1. Generate Synthetic Data
    # 252 days, Mean ~ 10% ann, Vol ~ 20% ann
    np.random.seed(42)
    periods = 252
    
    # Daily mean and vol
    mu_d = 0.10 / 252
    sigma_d = 0.20 / np.sqrt(252)
    
    returns = np.random.normal(mu_d, sigma_d, periods)
    
    # Introduce a big crash for Drawdown/VaR test
    # Day 100: -10% shock
    returns[100] = -0.10
    
    s_ret = pd.Series(returns)
    
    # 2. Compute Metrics
    engine = RiskEngine()
    metrics = engine.calculate(s_ret)
    
    print("\n[Scalar Metrics]")
    for k, v in metrics.items():
        print(f"{k:<20}: {v:.4f}")

    # Checks
    # Max Drawdown should be at least -10% due to shock (approx)
    if metrics['max_drawdown'] <= -0.09:
        print("✅ Max Drawdown captured shock (> 9%)")
    else:
        print(f"❌ Max Drawdown missed shock: {metrics['max_drawdown']}")
        
    # VaR 95% should be around -1.65 * sigma_d. 
    # sigma_d ~ 0.20/16 ~ 0.0125. VaR ~ -0.02.
    # However, we added a -0.10 outlier which is < 0.5% prob.
    # So 5% percentile should still be driven by normal noise roughly.
    print(f"Expected VaR (Parametric Normal): ~ {-1.645 * s_ret.std():.4f}")
    
    # 3. Rolling Metrics
    rolling = engine.rolling_metrics(s_ret, window=21) # 1 month
    print("\n[Rolling Metrics Head]")
    print(rolling.tail())
    
    if not rolling['drawdown'].isnull().all():
        print("✅ Rolling Drawdown computed")
    
    if not rolling['rolling_sharpe'].isnull().all():
        print("✅ Rolling Sharpe computed")

if __name__ == "__main__":
    main()
