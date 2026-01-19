import pandas as pd
import numpy as np
from src.features.engineering import TimeSeriesFeatures, StrictScaler, create_tabular_dataset, create_sequence_dataset
from src.config import settings

def main():
    # Load Canonical Data (assuming SPY exists from Phase 2)
    ticker = "SPY"
    path = settings.PROCESSED_DATA_DIR / f"{ticker}_canonical.parquet"
    
    if not path.exists():
        print(f"File not found: {path}")
        return

    df = pd.read_parquet(path)
    print(f"Loaded {len(df)} rows.")

    # 1. Generate Features
    print("\n--- Testing Feature Generation ---")
    X, y = create_tabular_dataset(df, lags=3)
    print("Features Head:")
    print(X.head(3))
    print(f"X Shape: {X.shape}, y Shape: {y.shape}")
    
    # Check for NaNs
    if X.isnull().values.any():
        print("❌ ERROR: NaNs found in X")
    else:
        print("✅ No NaNs in X")

    # 2. Test Strict Scaling (Leakage Check)
    print("\n--- Testing Strict Scaler ---")
    split_idx = int(len(X) * 0.8)
    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    
    scaler = StrictScaler()
    scaler.fit(X_train)
    
    print(f"Scaler Mean (Train fitted): {scaler.scaler.mean_}")
    
    X_test_scaled = scaler.transform(X_test)
    X_train_scaled = scaler.transform(X_train)
    
    # Verify manually that test data didn't affect mean
    # If we fit on ALL data, mean would be different.
    full_scaler = StrictScaler()
    full_scaler.fit(X)
    print(f"Global Mean (For comparison): {full_scaler.scaler.mean_}")
    
    if not np.allclose(scaler.scaler.mean_, full_scaler.scaler.mean_):
         print("✅ Scaler correctly ignores test data (Means differ)")
    else:
         print("⚠️ Warning: Means are identical (Dataset might be too small/stable or logic error)")

    # 3. Test LSTM Shaping
    print("\n--- Testing LSTM Sequence Shaping ---")
    seq_len = 5
    X_seq, y_seq = create_sequence_dataset(X_train_scaled, y.iloc[:split_idx], seq_length=seq_len)
    
    print(f"Sequence Shape: {X_seq.shape}")
    print(f"Target Shape: {y_seq.shape}")
    
    # Verification:
    # X_seq[0] should contain X_train_scaled rows 0 to 4
    # y_seq[0] should be y row 5
    print("\nVerifying Sequence alignment:")
    print("X_seq[0][-1] (Last step of first seq):") 
    print(X_seq[0][-1])
    print("X_train_scaled.iloc[seq_len-1] (Should match):")
    print(X_train_scaled.iloc[seq_len-1].values)
    
    if np.allclose(X_seq[0][-1], X_train_scaled.iloc[seq_len-1].values):
        print("✅ Sequence alignment correct")
    else:
        print("❌ Sequence alignment Mismatch")

if __name__ == "__main__":
    main()
