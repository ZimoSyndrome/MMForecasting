import pandas as pd
import numpy as np
from pandas.tseries.offsets import BDay
import logging
from typing import Optional

logger = logging.getLogger(__name__)

class DataProcessor:
    def __init__(self):
        # Fallback simplistic NYSE holidays if pandas_market_calendars is not used.
        # Ideally we'd use pandas_market_calendars, but for now we'll use BDay and basic filtering
        # or just rely on the intersection of data if we assume the source is somewhat reliable.
        # Given "robust" requirement, let's at least enforce business days.
        pass

    def clean_and_process(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """
        Takes raw OHLCV data and produces a canonical dataset:
        - Sort by date
        - Remove duplicates
        - Handle missing business days (forward fill max 5 days)
        - Calculate log returns and realized volatility
        - Filter out bad data (zero price, etc.)
        """
        if df.empty:
            logger.warning(f"Empty dataframe for {ticker}")
            return df

        # 1. Basic Cleaning
        df = df.sort_values('date').drop_duplicates(subset=['date'])
        df = df[df['close'] > 0].copy() # Remove zero/negative prices
        
        # 2. Date Alignment (Business Daily)
        df.set_index('date', inplace=True)
        # Ensure index is datetime
        df.index = pd.to_datetime(df.index)
        
        # Create a business day range from start to end
        full_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='B')
        
        # Reindex to catch missing business days
        original_len = len(df)
        df = df.reindex(full_range)
        
        # Forward fill small gaps (limit=5 days to avoid filling long delisted periods)
        df.ffill(limit=5, inplace=True)
        
        # Drop any remaining NaNs (e.g. at the start or if gaps > 5 days)
        processed_len = len(df)
        if processed_len > original_len:
             logger.info(f"Filled {processed_len - original_len} missing business days for {ticker}")

        df.dropna(inplace=True)

        # 3. Base Features
        # Log Returns: ln(P_t / P_{t-1})
        # Use Adjusted Close if available, else Close
        price_col = 'adj_close' if 'adj_close' in df.columns else 'close'
        df['log_return'] = np.log(df[price_col] / df[price_col].shift(1))
        
        # Realized Volatility (Annualized)
        # 21-day rolling standard deviation of log returns * sqrt(252)
        df['realized_vol'] = df['log_return'].rolling(window=21).std() * np.sqrt(252)

        # 4. Final Cleanup
        df.dropna(subset=['log_return', 'realized_vol'], inplace=True)
        
        # Reset index to make date a column again for standardized schema
        df.reset_index(inplace=True)
        df.rename(columns={'index': 'date'}, inplace=True)

        return df

if __name__ == "__main__":
    # Test stub
    pass
