import pandas as pd
import numpy as np
import logging
from src.features.engineering import TimeSeriesFeatures, create_tabular_dataset
from src.validation.validator import ChronologicalSplitter, WalkForwardValidator
from src.config import settings

# Dummy Model for testing the loop
class MeanRegressor:
    def __init__(self):
        self.mean = 0
        
    def fit(self, X, y):
        self.mean = y.mean()
        
    def predict(self, X):
        return np.full(len(X), self.mean)

def train_dummy_model(X_train, y_train):
    model = MeanRegressor()
    model.fit(X_train, y_train)
    return model

def predict_dummy_model(model, X_test):
    return model.predict(X_test)

def main():
    logging.basicConfig(level=logging.INFO)
    
    # 1. Load Data
    ticker = "SPY"
    path = settings.PROCESSED_DATA_DIR / f"{ticker}_canonical.parquet"
    if not path.exists():
        print("Data not found.")
        return
    df = pd.read_parquet(path)
    
    # 2. Prep Features
    X, y = create_tabular_dataset(df, lags=3)
    # Ensure index is datetime for validator
    X.index = pd.to_datetime(df.loc[X.index, 'date'])
    
    print(f"Total Data Points: {len(X)}")
    
    # 3. Setup Validator
    # Min train 30, test 1 step, expanding window (rolling=None)
    splitter = ChronologicalSplitter(min_train_size=30, test_size=1, step_size=1, rolling_window_size=None)
    validator = WalkForwardValidator(splitter)
    
    # 4. Run Loop
    print("\n--- Starting Verification Loop ---")
    results = validator.validate(X, y, train_dummy_model, predict_dummy_model)
    
    # 5. Check Output
    print("\n--- Results Head ---")
    print(results.head())
    print("\n--- Results Tail ---")
    print(results.tail())
    
    if len(results) == len(X) - 30:
        print(f"✅ Success: Generated {len(results)} forecasts (Expected {len(X)-30})")
    else:
        print(f"❌ Mismatch: Generated {len(results)} forecasts vs Expected {len(X)-30}")
        
    # Check if dates are unique and sorted
    if results['date'].is_monotonic_increasing:
        print("✅ Dates are monotonic increasing")
    else:
        print("❌ Dates are NOT monotonic")

    # Check leakage (simple check: training end date < test date)
    leakage_check = results[results['train_end_date'] >= results['date']]
    if leakage_check.empty:
        print("✅ No Temporal Leakage Detected (Train < Test)")
    else:
        print("❌ LEAKAGE DETECTED! Train End >= Test Date")
        print(leakage_check)

if __name__ == "__main__":
    main()
