import os
import logging
from datetime import datetime, timedelta
from typing import List, Optional

import pandas as pd
from binance.client import Client
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class HistoricalDataFetcher:
    def __init__(
        self,
        symbols: List[str],
        interval: str = Client.KLINE_INTERVAL_5MINUTE,
        days_back: int = 180,
        data_dir: str = "ml/data/raw"
    ):
        """Initialize the historical data fetcher.
        
        Args:
            symbols: List of trading pairs (e.g., ["BTCUSDT", "ETHUSDT"])
            interval: Kline interval (default: 5m)
            days_back: Number of days of historical data to fetch
            data_dir: Directory to save the data
        """
        self.symbols = symbols
        self.interval = interval
        self.days_back = days_back
        self.data_dir = data_dir
        
        # Initialize Binance client
        api_key = os.getenv("BINANCE_API_KEY")
        api_secret = os.getenv("BINANCE_API_SECRET")
        
        if not api_key or not api_secret:
            raise ValueError("BINANCE_API_KEY and BINANCE_API_SECRET must be set in .env file")
        
        self.client = Client(api_key, api_secret)
        
        # Create data directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)

    def fetch_klines(
        self,
        symbol: str,
        start_str: Optional[str] = None,
        end_str: Optional[str] = None
    ) -> pd.DataFrame:
        """Fetch klines (candlestick data) for a symbol.
        
        Args:
            symbol: Trading pair symbol
            start_str: Start time in UTC (default: days_back days ago)
            end_str: End time in UTC (default: now)
            
        Returns:
            DataFrame with OHLCV data
        """
        if not start_str:
            start_str = f"{self.days_back} day ago UTC"
        if not end_str:
            end_str = "now UTC"
            
        logger.info(f"Fetching {symbol} data from {start_str} to {end_str}")
        
        try:
            klines = self.client.get_historical_klines(
                symbol,
                self.interval,
                start_str,
                end_str
            )
            
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'num_trades',
                'taker_buy_base_volume', 'taker_buy_quote_volume', 'ignore'
            ])
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Select and rename columns
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            
            # Convert numeric columns
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {str(e)}")
            raise

    def save_data(self, df: pd.DataFrame, symbol: str) -> None:
        """Save DataFrame to CSV file.
        
        Args:
            df: DataFrame to save
            symbol: Trading pair symbol
        """
        filename = os.path.join(self.data_dir, f"{symbol}_{self.interval}.csv")
        df.to_csv(filename, index=False)
        logger.info(f"Saved data to {filename}")

    def fetch_all(self) -> None:
        """Fetch and save data for all symbols."""
        for symbol in self.symbols:
            try:
                df = self.fetch_klines(symbol)
                self.save_data(df, symbol)
            except Exception as e:
                logger.error(f"Failed to process {symbol}: {str(e)}")
                continue

def main():
    # Configuration
    SYMBOLS = ["BTCUSDT", "ETHUSDT", "ADAUSDT"]
    INTERVAL = Client.KLINE_INTERVAL_5MINUTE
    DAYS_BACK = 180
    DATA_DIR = "ml/data/raw"
    
    try:
        fetcher = HistoricalDataFetcher(
            symbols=SYMBOLS,
            interval=INTERVAL,
            days_back=DAYS_BACK,
            data_dir=DATA_DIR
        )
        fetcher.fetch_all()
        logger.info("Historical data fetching completed successfully")
    except Exception as e:
        logger.error(f"Failed to fetch historical data: {str(e)}")
        raise

if __name__ == "__main__":
    main() 