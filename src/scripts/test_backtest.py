import pandas as pd
import numpy as np
from src.backtest.engine import BacktestEngine, BacktestConfig

def main():
    print("--- Testing Backtest Engine ---")
    
    # 1. Create Mock Data
    # 5 days
    dates = pd.date_range(start="2024-01-01", periods=5)
    actuals =   [ 0.01, -0.01,  0.02, -0.02,  0.01]
    predicted = [ 0.005, -0.005, 0.005, -0.005, 0.005] # Perfect direction
    pred_vol =  [ 0.01,  0.01,  0.01,  0.02,  0.01] # Varying vol
    
    mock_df = pd.DataFrame({
        'date': dates,
        'actual': actuals,
        'predicted': predicted,
        'pred_vol': pred_vol
    })
    
    # 2. Test Basic Long-Only
    print("\n[Test 1] Long Only, No Costs")
    cfg = BacktestConfig(cost_bps=0.0, signal_logic="long_only")
    engine = BacktestEngine(cfg)
    res = engine.run(mock_df)
    
    print(res[['date', 'actual', 'signal', 'strat_ret_net', 'equity_curve']])
    
    # Check logic:
    # Row 0: pred > 0 -> Signal 1. Ret = 1 * 0.01 = 0.01. Equity = 10100.
    expected_eq = 10000 * 1.01
    if np.isclose(res.iloc[0]['equity_curve'], expected_eq):
        print("✅ Equity calc correct (Row 0)")
    else:
        print(f"❌ Equity mismatch: {res.iloc[0]['equity_curve']} vs {expected_eq}")

    # 3. Test Costs + Vol Target
    print("\n[Test 2] Vol Target (Target=16% ann ~ 1% daily), Costs 10bps")
    # Target daily = 0.16 / 16 ~ 0.01
    # Day 0: Vol=0.01. Lev = 1.0. W=1.0.
    # Day 3: Vol=0.02. Lev = 0.5. W=0.5 (if signal 1).
    
    cfg = BacktestConfig(cost_bps=0.0010, signal_logic="long_only", vol_target=0.16)
    engine = BacktestEngine(cfg)
    res = engine.run(mock_df)
    
    print(res[['date', 'signal', 'pred_vol', 'weight', 'turnover', 'cost', 'strat_ret_net']])
    
    # Check Weight Scaling at Day 3 (Vol 0.02)
    # Target ~0.01. Vol=0.02 -> Weight should be ~0.5.
    # Signal is -0.005 -> 0. Weight should be 0.
    
    # Let's check Day 0: Signal 1. Vol 0.01. Target 0.01. Weight ~ 1.0.
    w0 = res.iloc[0]['weight']
    if np.isclose(w0, 1.0, atol=0.1): 
        print(f"✅ Vol Target Weight correct (Vol=Match): {w0:.2f}")
    else:
        print(f"❌ Vol Target Weight Mismatch: {w0}")
        
    # Check Cost Calculation
    # Day 0: Prev W=0. W=1. Turnover=1. Cost=0.001.
    c0 = res.iloc[0]['cost']
    if np.isclose(c0, 0.001):
        print("✅ Cost correct (Row 0)")
    else:
        print(f"❌ Cost mismatch: {c0}")

if __name__ == "__main__":
    main()
