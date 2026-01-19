import pandas as pd
import logging
from src.benchmark.orchestrator import BenchmarkOrchestrator
from src.config import settings
from datetime import datetime

def main():
    # Setup Logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # 1. Define Universe (Reduced for Demo Speed)
    tickers = ["SPY", "QQQ", "GLD"]
    
    # 2. Define Period (2 Years: 1 Year Train + 1 Year Test)
    # Using fixed dates for reproducibility
    start_date = (datetime.now() - pd.DateOffset(years=2)).strftime("%Y-%m-%d")
    end_date = datetime.now().strftime("%Y-%m-%d")
    
    print(f"--- Starting Global Benchmark ---")
    print(f"Universe: {tickers}")
    print(f"Period: {start_date} to {end_date}")
    
    # 3. Initialize & Run
    orchestrator = BenchmarkOrchestrator(tickers, start_date, end_date, source="yahoo")
    df_metrics, df_global = orchestrator.run()
    
    # 4. Save Results
    results_dir = settings.DATA_DIR / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    if not df_metrics.empty:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save Detailed
        detailed_path = results_dir / f"benchmark_detailed_{timestamp}.csv"
        df_metrics.to_csv(detailed_path, index=False)
        print(f"\n✅ Detailed Results saved to: {detailed_path}")
        
        # Save Global Leaderboard
        leaderboard_path = results_dir / f"benchmark_leaderboard_{timestamp}.csv"
        df_global.to_csv(leaderboard_path)
        print(f"✅ Global Leaderboard saved to: {leaderboard_path}")
        
        # Print Leaderboard
        print("\n--- 🏆 GLOBAL LEADERBOARD (Sorted by Sharpe) ---")
        print(df_global[['annualized_return', 'annualized_vol', 'sharpe_ratio', 'max_drawdown']].round(4))
        
        # Print Best Per Ticker
        print("\n--- 🌟 BEST MODEL PER ASSET (Sharpe) ---")
        best_per_ticker = df_metrics.loc[df_metrics.groupby("Ticker")["sharpe_ratio"].idxmax()]
        print(best_per_ticker[['Ticker', 'Model', 'sharpe_ratio', 'annualized_return']].sort_values('sharpe_ratio', ascending=False))
        
    else:
        print("❌ Benchmark failed to generate results.")

if __name__ == "__main__":
    main()
