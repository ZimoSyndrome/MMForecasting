import xgboost as xgb
import numpy as np
import pandas as pd

class XGBoostModel:
    """
    ML Baseline: Gradient Boosting on Lagged Features
    """
    def __init__(self, params=None):
        self.params = params or {
            'n_estimators': 100,
            'max_depth': 3,
            'learning_rate': 0.1,
            'objective': 'reg:squarederror',
            'n_jobs': -1,
            'verbosity': 0
        }
        self.model = xgb.XGBRegressor(**self.params)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

# Interface for Validator
def train_xgboost(X_train, y_train):
    model = XGBoostModel()
    model.fit(X_train, y_train)
    return model

def predict_xgboost(model, X_test):
    # Returns (mean, nan) to match (mean, vol) signature if needed
    # Or simplified just mean. Validator handles tuple checks.
    pred = model.predict(X_test)
    n = len(pred)
    # Return (pred, None) since XGB doesn't predict vol explicitly here
    return (pred, np.full(n, np.nan))
