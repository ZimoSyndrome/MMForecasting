import pandas as pd
import numpy as np
import logging
import warnings
from src.features.engineering import create_tabular_dataset
from src.validation.validator import ChronologicalSplitter, WalkForwardValidator
from src.config import settings

# Lazy imports to prevent crash on missing deps
try:
    from src.models.arima import train_arima_garch, predict_arima_garch
except Exception as e:
    print(f"⚠️ ARIMA model unavailable: {e}")
    train_arima_garch = None
    predict_arima_garch = None

try:
    from src.models.xgb import train_xgboost, predict_xgboost
except Exception as e:
    print(f"⚠️ XGBoost model unavailable: {e}")
    train_xgboost = None
    predict_xgboost = None

try:
    from src.models.lstm import train_lstm, predict_lstm
except Exception as e:
    print(f"⚠️ LSTM model unavailable: {e}")
    train_lstm = None
    predict_lstm = None

# Suppress warnings for clean output
warnings.filterwarnings("ignore")

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    logger = logging.getLogger(__name__)

    # 1. Load Data
    ticker = "SPY"
    path = settings.PROCESSED_DATA_DIR / f"{ticker}_canonical.parquet"
    if not path.exists():
        logger.error("Data not found.")
        return
        
    df = pd.read_parquet(path)
    # Convert to standard X, y
    X, y = create_tabular_dataset(df, lags=3)
    X.index = pd.to_datetime(df.loc[X.index, 'date'])
    
    logger.info(f"Loaded Data: {len(X)} samples.")

    # 2. Setup Validation Engine
    # Small window for testing speed
    splitter = ChronologicalSplitter(min_train_size=40, test_size=1, step_size=1)
    validator = WalkForwardValidator(splitter)

    models = {
        "ARIMA-GARCH": (train_arima_garch, predict_arima_garch),
        "XGBoost": (train_xgboost, predict_xgboost),
        "LSTM": (train_lstm, predict_lstm)
    }
    
    # Filter out missing models
    models = {k: v for k, v in models.items() if v[0] is not None}

    # 3. Run Validation for each model
    for name, (trainer, predictor) in models.items():
        print(f"\n⚡️ Testing Model: {name}")
        try:
            results = validator.validate(X, y, trainer, predictor)
            
            print(f"✅ {name} Completed. Rows: {len(results)}")
            print(results[['date', 'actual', 'predicted', 'pred_vol']].head(3))
            
            # Basic sanity checks
            if results['predicted'].isnull().any():
                print(f"❌ {name} produced NaNs in predictions!")
            
            if name == "ARIMA-GARCH":
                if results['pred_vol'].isnull().any() or (results['pred_vol'] == 0).all():
                     print(f"⚠️ {name} Volatility Forecasts might be invalid (All NaNs or Zeros).")
                else:
                     print(f"✅ {name} Volatility Forecasts OK.")
                     
        except Exception as e:
            print(f"❌ {name} Failed: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
