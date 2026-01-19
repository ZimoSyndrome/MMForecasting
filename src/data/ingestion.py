import pandas as pd
import yfinance as yf
from alpaca_trade_api.rest import REST, TimeFrame
from datetime import datetime, date
from typing import Optional, Union
import logging

from src.config import settings

logger = logging.getLogger(__name__)

class DataIngestion:
    def __init__(self):
        self.alpaca = None
        if settings.ALPACA_API_KEY and settings.ALPACA_SECRET_KEY:
            try:
                self.alpaca = REST(
                    settings.ALPACA_API_KEY,
                    settings.ALPACA_SECRET_KEY,
                    base_url=settings.ALPACA_ENDPOINT
                )
            except Exception as e:
                logger.warning(f"Failed to initialize Alpaca API: {e}")
        else:
            logger.warning("Alpaca credentials not found. Alpaca fetching will be disabled.")

    def fetch_data(
        self, 
        ticker: str, 
        start_date: Union[str, date, datetime], 
        end_date: Union[str, date, datetime], 
        source: str = "alpaca",
        force_fresh: bool = False
    ) -> pd.DataFrame:
        """
        Fetch market data for a ticker.
        """
        # Ensure dates are strings YYYY-MM-DD
        start_str = self._format_date(start_date)
        end_str = self._format_date(end_date)
        
        # Check if local file exists
        file_path = settings.RAW_DATA_DIR / f"{ticker}_{source}.parquet"
        
        if file_path.exists() and not force_fresh:
            logger.info(f"Loading cached data for {ticker} from {file_path}")
            df = pd.read_parquet(file_path)
            # Basic validation: check if cached data covers the requested range?
            # For simplicity, if cached, specific logic might be needed to merge or refill.
            # Here we just return cached if available, but for a real system we might want to append.
            # User can use force_fresh=True to reload.
            return df

        if source == "alpaca":
            df = self._fetch_alpaca(ticker, start_str, end_str)
        elif source == "yahoo":
            df = self._fetch_yahoo(ticker, start_str, end_str)
        else:
            raise ValueError(f"Unknown source: {source}")
            
        if not df.empty:
            logger.info(f"Saving raw data for {ticker} to {file_path}")
            df.to_parquet(file_path)
            
        return df

    def _fetch_alpaca(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        if not self.alpaca:
            logger.error("Alpaca API not initialized.")
            return pd.DataFrame()
            
        logger.info(f"Fetching {ticker} from Alpaca ({start} to {end})...")
        try:
            bars = self.alpaca.get_bars(
                ticker, 
                TimeFrame.Day, 
                start=start, 
                end=end, 
                adjustment='raw',
                feed='iex'  # Required for free/basic tier
            ).df
            
            if bars.empty:
                logger.warning(f"No data returned from Alpaca for {ticker}")
                return bars

            # Normalize columns
            # Alpaca returns: open, high, low, close, volume, trade_count, vwap
            # We want: open, high, low, close, volume, (adj_close not strictly provided by raw bars)
            bars = bars.reset_index() # timestamp is index
            
            # Rename columns to match schema standard (lowercase)
            bars.rename(columns={'timestamp': 'date'}, inplace=True)
            
            # Ensure UTC and remove timezone for parquet compatibility issues sometimes
            if 'date' in bars.columns:
                 bars['date'] = pd.to_datetime(bars['date']).dt.tz_convert(None)

            return bars
        except Exception as e:
            logger.error(f"Error fetching from Alpaca: {e}")
            return pd.DataFrame()

    def _fetch_yahoo(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        logger.info(f"Fetching {ticker} from Yahoo Finance ({start} to {end})...")
        try:
            df = yf.download(ticker, start=start, end=end, progress=False)
            
            if df.empty:
                logger.warning(f"No data returned from Yahoo for {ticker}")
                return df
                
            # Handle MultiIndex columns if present
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            # Reset index to make Date a column
            df.reset_index(inplace=True)

            # Standardize columns to lowercase (Including 'Date' -> 'date')
            df.columns = [c.lower() for c in df.columns] 
            
            # Rename columns
            rename_map = {
                'adj close': 'adj_close'
            }
            df.rename(columns=rename_map, inplace=True)
             
            # Standardize columns
            expected_cols = ['date', 'open', 'high', 'low', 'close', 'volume', 'adj_close']
            df = df[[c for c in expected_cols if c in df.columns]]
            
            return df
        except Exception as e:
            logger.error(f"Error fetching from Yahoo: {e}")
            return pd.DataFrame()

    def _format_date(self, d: Union[str, date, datetime]) -> str:
        if isinstance(d, datetime) or isinstance(d, date):
            return d.strftime("%Y-%m-%d")
        return d

if __name__ == "__main__":
    # Simple test
    logging.basicConfig(level=logging.INFO)
    ingest = DataIngestion()
    # ingestion.fetch_data("AAPL", "2023-01-01", "2023-01-31", source="yahoo")
