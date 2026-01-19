import argparse
import logging
from datetime import datetime, timedelta
from src.data.ingestion import DataIngestion

def main():
    parser = argparse.ArgumentParser(description="Fetch market data for a ticker.")
    parser.add_argument("--ticker", type=str, required=True, help="Ticker symbol (e.g. SPY)")
    parser.add_argument("--start", type=str, default=(datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d"), help="Start date YYYY-MM-DD")
    parser.add_argument("--end", type=str, default=datetime.now().strftime("%Y-%m-%d"), help="End date YYYY-MM-DD")
    parser.add_argument("--source", type=str, default="yahoo", choices=["alpaca", "yahoo"], help="Data source")
    parser.add_argument("--force", action="store_true", help="Force refresh even if local file exists")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    ingest = DataIngestion()
    df = ingest.fetch_data(args.ticker, args.start, args.end, source=args.source, force_fresh=args.force)
    
    if not df.empty:
        print(f"Successfully fetched {len(df)} rows for {args.ticker}")
        print(df.head())
        print(df.tail())
    else:
        print(f"Failed to fetch data for {args.ticker}")

if __name__ == "__main__":
    main()
