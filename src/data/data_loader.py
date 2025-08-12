"""
Data loader for fetching market data from Yahoo Finance.
Handles data downloading, caching, and preprocessing.
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
import os
import pickle


class DataLoader:
    """
    Handles downloading and caching of market data from Yahoo Finance.
    """
    
    def __init__(self, cache_dir: str = "data/cache"):
        """
        Initialize data loader.
        
        Args:
            cache_dir: Directory to store cached data
        """
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    def get_data(self, 
                 symbol: str, 
                 start_date: str, 
                 end_date: str,
                 interval: str = "1d",
                 use_cache: bool = True) -> pd.DataFrame:
        """
        Get market data for a symbol.
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            interval: Data interval (1d, 1h, etc.)
            use_cache: Whether to use cached data if available
            
        Returns:
            DataFrame with OHLCV data
        """
        cache_key = f"{symbol}_{start_date}_{end_date}_{interval}.pkl"
        cache_path = os.path.join(self.cache_dir, cache_key)
        
        # Try to load from cache first
        if use_cache and os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    data = pickle.load(f)
                print(f"Loaded {symbol} data from cache")
                return data
            except Exception as e:
                print(f"Cache loading failed: {e}")
        
        # Download fresh data
        print(f"Downloading {symbol} data from Yahoo Finance...")
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start_date, end=end_date, interval=interval)
            
            if data.empty:
                raise ValueError(f"No data found for {symbol}")
            
            # Cache the data
            if use_cache:
                try:
                    with open(cache_path, 'wb') as f:
                        pickle.dump(data, f)
                    print(f"Cached {symbol} data")
                except Exception as e:
                    print(f"Caching failed: {e}")
            
            return data
            
        except Exception as e:
            raise Exception(f"Failed to download data for {symbol}: {e}")
    
    def get_multiple_symbols(self, 
                            symbols: List[str], 
                            start_date: str, 
                            end_date: str,
                            interval: str = "1d") -> Dict[str, pd.DataFrame]:
        """
        Get data for multiple symbols.
        
        Args:
            symbols: List of stock symbols
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            interval: Data interval
            
        Returns:
            Dictionary mapping symbols to DataFrames
        """
        data_dict = {}
        
        for symbol in symbols:
            try:
                data_dict[symbol] = self.get_data(symbol, start_date, end_date, interval)
            except Exception as e:
                print(f"Warning: Failed to get data for {symbol}: {e}")
                continue
        
        return data_dict
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """
        Validate that data has required columns and no major gaps.
        
        Args:
            data: DataFrame to validate
            
        Returns:
            True if data is valid
        """
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        # Check required columns
        if not all(col in data.columns for col in required_columns):
            return False
        
        # Check for excessive missing data
        missing_pct = data.isnull().sum().sum() / (len(data) * len(required_columns))
        if missing_pct > 0.1:  # More than 10% missing
            return False
        
        # Check for reasonable price ranges
        if (data['Close'] <= 0).any():
            return False
        
        return True
    
    def get_data_info(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Get summary information about the data.
        
        Args:
            data: DataFrame to analyze
            
        Returns:
            Dictionary with data summary
        """
        return {
            'start_date': data.index[0],
            'end_date': data.index[-1],
            'total_days': len(data),
            'missing_data': data.isnull().sum().to_dict(),
            'price_range': {
                'min_close': data['Close'].min(),
                'max_close': data['Close'].max(),
                'avg_close': data['Close'].mean()
            },
            'volume_stats': {
                'total_volume': data['Volume'].sum(),
                'avg_volume': data['Volume'].mean()
            }
        }
