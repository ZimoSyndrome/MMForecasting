from typing import Iterator, Tuple, List, Optional, Union, Dict, Any
import pandas as pd
import numpy as np
from dataclasses import dataclass
import logging
from tqdm import tqdm
from src.features.engineering import StrictScaler

logger = logging.getLogger(__name__)

@dataclass
class ValidationSplit:
    fold_id: int
    train_indices: np.ndarray
    test_indices: np.ndarray

class ChronologicalSplitter:
    """
    Custom splitter for rolling/expanding windows.
    Does NOT shuffle. Respects time order.
    """
    def __init__(
        self, 
        min_train_size: int, 
        test_size: int = 1, 
        step_size: int = 1, 
        rolling_window_size: Optional[int] = None
    ):
        self.min_train_size = min_train_size
        self.test_size = test_size
        self.step_size = step_size
        self.rolling_window_size = rolling_window_size

    def split(self, X: Union[pd.DataFrame, np.ndarray]) -> Iterator[ValidationSplit]:
        n_samples = len(X)
        indices = np.arange(n_samples)
        
        # Start looking for splits
        # We need at least min_train_size samples before we can test
        # current_test_start is the index of the first test sample
        current_test_start = self.min_train_size
        
        fold_id = 0
        while current_test_start + self.test_size <= n_samples:
            test_end = current_test_start + self.test_size
            test_idx = indices[current_test_start : test_end]
            
            # Determine Train Range
            train_end = current_test_start
            
            if self.rolling_window_size:
                # Rolling: use last N samples defined by rolling_window_size
                train_start = max(0, train_end - self.rolling_window_size)
            else:
                # Expanding: start from 0
                train_start = 0
                
            train_idx = indices[train_start : train_end]
            
            yield ValidationSplit(fold_id, train_idx, test_idx)
            
            fold_id += 1
            current_test_start += self.step_size

class WalkForwardValidator:
    """
    The Referee. Enforces strictly causal validation loop.
    1. Split
    2. Fit Scaler (Train Only)
    3. Transform (Train & Test)
    4. Train Model (Train Only) -> Caller provided function
    5. Predict (Test Only)
    """
    def __init__(self, splitter: ChronologicalSplitter):
        self.splitter = splitter

    def validate(
        self, 
        X: pd.DataFrame, 
        y: pd.Series, 
        model_trainer: Any, # Function/Callable that takes (X_train, y_train) and returns fitted model
        model_predictor: Any # Function/Callable that takes (fitted_model, X_test) and returns preds
    ) -> pd.DataFrame:
        
        results = []
        
        # Generator for splits
        splits = list(self.splitter.split(X))
        logger.info(f"Starting Walk-Forward Validation: {len(splits)} folds")
        
        for split in tqdm(splits, desc="Walk-Forward Folds"):
            # 1. Slice Data
            X_train_raw = X.iloc[split.train_indices]
            y_train = y.iloc[split.train_indices]
            
            X_test_raw = X.iloc[split.test_indices]
            y_test = y.iloc[split.test_indices] # We only use this for "Actual", not for prediction!
            
            # 2. Strict Scaling
            scaler = StrictScaler()
            scaler.fit(X_train_raw)
            
            X_train = scaler.transform(X_train_raw)
            X_test = scaler.transform(X_test_raw)
            
            # 3. Train Model
            # Note: We pass X_train (scaled) and y_train
            model = model_trainer(X_train, y_train)
            
            # 4. Predict
            # Note: We pass X_test (scaled)
            # Returns tuple (pred, pred_vol) or just pred
            preds = model_predictor(model, X_test)
            
            # Handle different return types (just mean, or mean+vol)
            pred_mean = preds[0] if isinstance(preds, tuple) else preds
            pred_vol = preds[1] if isinstance(preds, tuple) and len(preds) > 1 else np.nan
            
            # 5. Store Result
            # Assuming test_size=1 used usually, but loop handles >1
            for i, idx in enumerate(split.test_indices):
                results.append({
                    "date": X.index[idx], # Assuming index is datetime
                    "actual": y_test.iloc[i],
                    "predicted": pred_mean[i] if hasattr(pred_mean, '__getitem__') else pred_mean,
                    "pred_vol": pred_vol[i] if hasattr(pred_vol, '__getitem__') and not np.isnan(pred_vol) else None,
                    "fold_id": split.fold_id,
                    "train_end_date": X.index[split.train_indices[-1]]
                })
                
        return pd.DataFrame(results)

if __name__ == "__main__":
    pass
