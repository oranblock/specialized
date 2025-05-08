#!/usr/bin/env python
"""
Forex Data Collector for Multi-Component Pattern Prediction System

This module handles collecting raw price data from Polygon.io and storing it
in a structured format for further analysis.
"""

import os
import sys
import argparse
import logging
import json
import time
from datetime import datetime, timedelta
import pandas as pd
import requests
import signal
import atexit

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("data_collector.log")
    ]
)
logger = logging.getLogger("DataCollector")

class ForexDataCollector:
    """Collects and stores forex price data from Polygon.io API."""
    
    def __init__(self, api_key, symbol, timeframe='5m', data_dir=None):
        """
        Initialize the data collector.
        
        Args:
            api_key: Polygon.io API key
            symbol: Currency pair (format: XXX_YYY, e.g., EUR_USD)
            timeframe: Data timeframe (default: 5m)
            data_dir: Directory to store collected data
        """
        self.api_key = api_key
        self.symbol = symbol
        self.timeframe = timeframe
        
        # Parse currency pair
        self.from_currency, self.to_currency = symbol.split('_')
        
        # Set up data storage
        if data_dir is None:
            self.data_dir = os.path.abspath(os.path.join(
                os.path.dirname(__file__), '..', '..', 'data', 'raw'
            ))
        else:
            self.data_dir = os.path.abspath(data_dir)
            
        # Ensure data directory exists
        self._ensure_directory(self.data_dir)
        self._ensure_directory(os.path.join(self.data_dir, self.symbol))
        self._ensure_directory(os.path.join(self.data_dir, self.symbol, self.timeframe))
        
        # Initialize data storage
        self.data = []
        self.last_save_time = datetime.now()
        self.collection_start_time = datetime.now()
        self.running = False
        self.request_count = 0
        self.error_count = 0
        self.reconnect_attempts = 0
        self.last_price = None
        
        # Register shutdown handlers
        atexit.register(self.shutdown)
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info(f"Initialized data collector for {symbol} with {timeframe} timeframe")
        logger.info(f"Data will be stored in {self.data_dir}")
    
    def _ensure_directory(self, directory):
        """Ensure directory exists, create if needed."""
        if not os.path.exists(directory):
            os.makedirs(directory)
            logger.info(f"Created directory: {directory}")
    
    def _signal_handler(self, sig, frame):
        """Handle termination signals."""
        logger.info(f"Received signal {sig}, shutting down...")
        self.shutdown()
        sys.exit(0)
        
    def _get_data_filename(self):
        """Generate filename for the current data collection."""
        date_str = datetime.now().strftime('%Y%m%d')
        return os.path.join(
            self.data_dir, 
            self.symbol, 
            self.timeframe,
            f"{date_str}_{self.symbol}_{self.timeframe}.csv"
        )
        
    def _get_metadata_filename(self):
        """Generate filename for the metadata."""
        date_str = datetime.now().strftime('%Y%m%d')
        return os.path.join(
            self.data_dir, 
            self.symbol, 
            self.timeframe,
            f"{date_str}_{self.symbol}_{self.timeframe}_metadata.json"
        )
    
    def _save_data(self, force=False):
        """
        Save collected data to CSV file.
        
        Args:
            force: Force save even if save interval not reached
        """
        now = datetime.now()
        # Save every 5 minutes or if forced
        if force or (now - self.last_save_time).total_seconds() >= 300:
            if not self.data:
                logger.warning("No data to save")
                return
                
            # Convert data to DataFrame
            df = pd.DataFrame(self.data)
            
            # Ensure timestamp is datetime
            if 'timestamp' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Sort by timestamp
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            # Get filename
            filename = self._get_data_filename()
            
            # If file exists, append without header
            mode = 'a' if os.path.exists(filename) else 'w'
            header = not os.path.exists(filename)
            
            # Save to CSV
            df.to_csv(filename, mode=mode, header=header, index=False)
            
            # Save metadata
            metadata = {
                'symbol': self.symbol,
                'timeframe': self.timeframe,
                'collection_start': self.collection_start_time.isoformat(),
                'last_update': now.isoformat(),
                'records_collected': len(self.data),
                'total_records': len(df) if mode == 'w' else None,
                'request_count': self.request_count,
                'error_count': self.error_count
            }
            
            with open(self._get_metadata_filename(), 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Saved {len(self.data)} records to {filename}")
            
            # Clear data after saving
            self.data = []
            self.last_save_time = now
    
    def fetch_current_price(self):
        """
        Fetch current price from Polygon.io API.
        
        Returns:
            dict: Candle data if successful, None otherwise
        """
        try:
            # Increment request counter
            self.request_count += 1
            
            # Build URL for REST API
            url = f"https://api.polygon.io/v1/last_quote/currencies/{self.from_currency}/{self.to_currency}?apiKey={self.api_key}"
            
            # Make request
            response = requests.get(url, timeout=10)
            
            # Check if request was successful
            if response.status_code == 200:
                data = response.json()
                
                # Reset error counter on success
                self.error_count = 0
                self.reconnect_attempts = 0
                
                # Extract price data
                if 'last' in data and isinstance(data['last'], dict):
                    bid = float(data['last'].get('bid', 0))
                    ask = float(data['last'].get('ask', 0))
                    
                    # Calculate mid price
                    price = (bid + ask) / 2
                    
                    # Get current timestamp
                    timestamp = datetime.now()
                    
                    # Create candle data
                    candle = {
                        'timestamp': timestamp,
                        'open': price,
                        'high': price,
                        'low': price,
                        'close': price,
                        'volume': 0,
                        'bid': bid,
                        'ask': ask
                    }
                    
                    # Log if price changed significantly
                    if self.last_price is not None:
                        change = abs(price - self.last_price) / self.last_price * 100
                        if change > 0.05:  # Log if >0.05% change
                            logger.info(f"Price changed by {change:.4f}%: {self.last_price:.5f} -> {price:.5f}")
                    
                    self.last_price = price
                    return candle
                else:
                    logger.warning(f"Unexpected response format: {data}")
                    return None
            else:
                self.error_count += 1
                logger.error(f"API request failed with status {response.status_code}: {response.text}")
                return None
                
        except Exception as e:
            self.error_count += 1
            logger.error(f"Error fetching price: {str(e)}")
            return None
    
    def start_collection(self, interval_seconds=5, max_runtime=None):
        """
        Start collecting data at specified interval.
        
        Args:
            interval_seconds: Seconds between API calls
            max_runtime: Maximum runtime in seconds (None for indefinite)
        """
        self.running = True
        logger.info(f"Starting data collection for {self.symbol} at {interval_seconds}s intervals")
        
        start_time = datetime.now()
        self.collection_start_time = start_time
        
        try:
            while self.running:
                # Check max runtime
                if max_runtime is not None:
                    elapsed = (datetime.now() - start_time).total_seconds()
                    if elapsed >= max_runtime:
                        logger.info(f"Reached maximum runtime of {max_runtime}s")
                        break
                
                # Fetch data
                candle = self.fetch_current_price()
                
                # Store data if available
                if candle is not None:
                    self.data.append(candle)
                    
                    # Log occasional updates
                    if len(self.data) % 10 == 0:
                        logger.info(f"Collected {len(self.data)} data points")
                    
                    # Save periodically
                    self._save_data()
                
                # Handle errors with backoff
                elif self.error_count > 0 and self.error_count % 3 == 0:
                    backoff = min(60, 5 * 2 ** min(self.reconnect_attempts, 6))  # Max 60s backoff
                    self.reconnect_attempts += 1
                    logger.warning(f"Backing off for {backoff}s after {self.error_count} errors")
                    time.sleep(backoff)
                    continue
                
                # Wait for next interval
                time.sleep(interval_seconds)
                
        except KeyboardInterrupt:
            logger.info("Data collection interrupted by user")
        finally:
            self.shutdown()
    
    def shutdown(self):
        """Shutdown collector and save any remaining data."""
        if self.running:
            self.running = False
            logger.info("Shutting down data collector...")
            self._save_data(force=True)
            
            # Save final metadata
            collection_end_time = datetime.now()
            duration = (collection_end_time - self.collection_start_time).total_seconds()
            
            metadata = {
                'symbol': self.symbol,
                'timeframe': self.timeframe,
                'collection_start': self.collection_start_time.isoformat(),
                'collection_end': collection_end_time.isoformat(),
                'duration_seconds': duration,
                'request_count': self.request_count,
                'error_count': self.error_count
            }
            
            metadata_file = self._get_metadata_filename().replace('.json', '_final.json')
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Data collection completed in {duration:.1f}s")
            logger.info(f"Made {self.request_count} API requests with {self.error_count} errors")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Forex Data Collector')
    parser.add_argument('--api-key', required=True, help='Polygon.io API key')
    parser.add_argument('--symbol', default='EUR_USD', help='Forex pair symbol (e.g., EUR_USD)')
    parser.add_argument('--timeframe', default='5m', help='Data timeframe')
    parser.add_argument('--interval', type=int, default=5, help='Polling interval in seconds')
    parser.add_argument('--data-dir', help='Directory to store data')
    parser.add_argument('--runtime', type=int, help='Maximum runtime in seconds')
    return parser.parse_args()

def main():
    """Main function."""
    args = parse_args()
    
    collector = ForexDataCollector(
        api_key=args.api_key,
        symbol=args.symbol,
        timeframe=args.timeframe,
        data_dir=args.data_dir
    )
    
    collector.start_collection(
        interval_seconds=args.interval,
        max_runtime=args.runtime
    )

if __name__ == "__main__":
    main()