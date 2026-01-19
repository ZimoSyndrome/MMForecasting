import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
from datetime import datetime, timedelta

from src.data.ingestion import DataIngestion
from src.data.processing import DataProcessor
from src.features.engineering import create_tabular_dataset
from src.validation.validator import ChronologicalSplitter, WalkForwardValidator
from src.backtest.engine import BacktestEngine, BacktestConfig
from src.risk.metrics import RiskEngine


logger = logging.getLogger(__name__)

# Import Models (Robustly)
try:
    from src.models.arima import train_arima_garch, predict_arima_garch
except Exception as e:
    logger.warning(f"ARIMA unavailable: {e}")
    train_arima_garch = None

try:
    from src.models.xgb import train_xgboost, predict_xgboost
except Exception as e:
    logger.warning(f"XGBoost unavailable: {e}")
    train_xgboost = None

try:
    from src.models.lstm import train_lstm, predict_lstm
except Exception as e:
    logger.warning(f"LSTM unavailable: {e}")
    train_lstm = None

class BenchmarkOrchestrator:
    def __init__(self, tickers: List[str], start_date: str, end_date: str, source: str = "yahoo"):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.source = source
        
        # Define available models
        self.models = {}
        if train_arima_garch:
            self.models["ARIMA-GARCH"] = (train_arima_garch, predict_arima_garch)
        if train_xgboost:
            self.models["XGBoost"] = (train_xgboost, predict_xgboost)
        if train_lstm:
            self.models["LSTM"] = (train_lstm, predict_lstm)
            
        self.full_results = []
        self.metrics_summary = []

    def run(self):
        """
        Main execution loop.
        """
        logger.info(f"Starting Benchmark for {len(self.tickers)} tickers: {self.tickers}")
        
        for ticker in tqdm(self.tickers, desc="Processing Tickers"):
            try:
                self._process_ticker(ticker)
            except Exception as e:
                logger.error(f"Failed to process {ticker}: {e}")
                import traceback
                traceback.print_exc()

        return self._generate_reports()

    def _process_ticker(self, ticker: str):
        # 1. Ingestion
        ingest = DataIngestion()
        df_raw = ingest.fetch_data(ticker, self.start_date, self.end_date, source=self.source)
        if df_raw.empty:
            logger.warning(f"No data for {ticker}")
            return

        # 2. Processing
        processor = DataProcessor()
        df_clean = processor.clean_and_process(df_raw, ticker) # Returns df with log_return, realized_vol
        
        # 3. Features
        # Using 3 lags as standard
        X, y = create_tabular_dataset(df_clean, lags=3)
        # Ensure DateTime index
        X.index = pd.to_datetime(df_clean.loc[X.index, 'date'])
        
        if len(X) < 100:
            logger.warning(f"Insufficient data points for {ticker}: {len(X)}")
            return

        # 4. Validation Setup
        # Use expanding window or rolling?
        # Let's use Expanding window (rolling=None) for stability in benchmark
        # Min train = 1 year ~ 252 days
        splitter = ChronologicalSplitter(min_train_size=252, test_size=1, step_size=1)
        validator = WalkForwardValidator(splitter)
        
        # 5. Run Models
        for model_name, (trainer, predictor) in self.models.items():
            logger.info(f"  Running {model_name} for {ticker}...")
            
            try:
                # A. Walk-Forward Validation
                # Helper to catch errors inside loop
                res_df = validator.validate(X, y, trainer, predictor)
                
                if res_df.empty:
                    continue
                
                # B. Backtest
                # Long-Only, 10bps cost
                bt_cfg = BacktestConfig(cost_bps=0.0010, signal_logic="long_only")
                bt_engine = BacktestEngine(bt_cfg)
                bt_res = bt_engine.run(res_df)
                
                # C. Risk Metrics
                risk_engine = RiskEngine()
                metrics = risk_engine.calculate(bt_res['strat_ret_net'])
                
                # Store aggregated metrics
                metric_row = {
                    "Ticker": ticker,
                    "Model": model_name,
                    "Samples": len(bt_res),
                    **metrics
                }
                self.metrics_summary.append(metric_row)
                
                # Store detailed time-series (optional, can be heavy)
                # Just keep necessary cols
                bt_res['Ticker'] = ticker
                bt_res['Model'] = model_name
                self.full_results.append(bt_res)
                
            except Exception as e:
                logger.error(f"Error in {model_name} for {ticker}: {e}")

    def _generate_reports(self):
        # 1. Metrics DataFrame
        df_metrics = pd.DataFrame(self.metrics_summary)
        
        if df_metrics.empty:
            return df_metrics, pd.DataFrame()

        # 2. Global Leaderboard (Average across tickers)
        # Select numeric columns only
        numeric_cols = df_metrics.select_dtypes(include=[np.number]).columns
        # Group by Model
        df_global = df_metrics.groupby("Model")[numeric_cols].mean().sort_values("sharpe_ratio", ascending=False)
        
        return df_metrics, df_global
