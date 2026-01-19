import argparse
import logging
from src.data.ingestion import DataIngestion
from src.data.processing import DataProcessor
from src.config import settings

def main():
    parser = argparse.ArgumentParser(description="Create canonical dataset for a ticker.")
    parser.add_argument("--ticker", type=str, required=True, help="Ticker symbol (e.g. SPY)")
    parser.add_argument("--start", type=str, default="2020-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", type=str, default="2024-01-01", help="End date YYYY-MM-DD")
    # Default to yahoo for stability in verification if alpaca fails, but code prefers alpaca if keys exist
    parser.add_argument("--source", type=str, default=settings.DEFAULT_DATA_SOURCE, help="Data source") 
    parser.add_argument("--force", action="store_true", help="Force fresh fetch")

    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    # 1. Ingest
    ingester = DataIngestion()
    raw_df = ingester.fetch_data(args.ticker, args.start, args.end, source=args.source, force_fresh=args.force)
    
    if raw_df.empty:
        logger.error("Ingestion failed. Exiting.")
        return

    # 2. Process
    processor = DataProcessor()
    clean_df = processor.clean_and_process(raw_df, args.ticker)
    
    if clean_df.empty:
        logger.error("Processing resulted in empty dataframe.")
        return

    # 3. Save Canonical
    output_path = settings.PROCESSED_DATA_DIR / f"{args.ticker}_canonical.parquet"
    if not output_path.parent.exists():
        output_path.parent.mkdir(parents=True)
        
    clean_df.to_parquet(output_path)
    logger.info(f"Saved canonical dataset to {output_path}")
    
    # 4. Show Stats
    print("\n--- Canonical Dataset Stats ---")
    print(f"Rows: {len(clean_df)}")
    print(f"Date Range: {clean_df['date'].min()} to {clean_df['date'].max()}")
    print("\nTail:")
    print(clean_df.tail())
    print("\nColumns:")
    print(clean_df.columns.tolist())

if __name__ == "__main__":
    main()
