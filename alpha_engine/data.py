"""
Data ingestion and processing module for the Alpha Engine.

This module handles downloading, cleaning, and preprocessing of equity data
from Yahoo Finance and other sources.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from pathlib import Path
import yaml
from typing import List, Dict, Optional, Tuple
import logging
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    """
    Handles data downloading and initial processing from various sources.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize DataLoader with configuration.
        
        Args:
            config_path: Path to configuration YAML file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.data_config = self.config['data']
        self.data_path = Path(self.data_config['data_path'])
        self.data_path.mkdir(parents=True, exist_ok=True)
        
    def get_universe_tickers(self, universe: str = None) -> List[str]:
        """
        Get list of tickers for specified universe.
        
        Args:
            universe: Universe name (sp500, sp1500, russell3000)
            
        Returns:
            List of ticker symbols
        """
        if universe is None:
            universe = self.data_config['universe']
            
        # For this implementation, we'll use a representative sample
        # In production, you'd fetch from actual index composition APIs
        if universe == "sp500":
            # Sample S&P 500 tickers - in production get from actual source
            tickers = [
                'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'BRK-B',
                'UNH', 'JNJ', 'XOM', 'JPM', 'V', 'PG', 'MA', 'HD', 'CVX', 'LLY',
                'ABBV', 'PFE', 'BAC', 'KO', 'AVGO', 'PEP', 'TMO', 'WMT', 'COST',
                'DIS', 'ABT', 'MRK', 'ACN', 'VZ', 'DHR', 'NFLX', 'ADBE', 'TXN',
                'NKE', 'LIN', 'QCOM', 'PM', 'T', 'RTX', 'LOW', 'UPS', 'NEE',
                'HON', 'SPGI', 'IBM', 'GS', 'MDT', 'INTU', 'CAT', 'AMGN', 'ISRG',
                'BKNG', 'AXP', 'GE', 'DE', 'MMM', 'TJX', 'MU', 'AMD', 'BLK',
                'SYK', 'AMAT', 'GILD', 'CVS', 'C', 'MDLZ', 'VRTX', 'ADI', 'TMUS',
                'NOW', 'PYPL', 'ZTS', 'CI', 'REGN', 'CB', 'SLB', 'MO', 'EQIX',
                'DUK', 'CL', 'SO', 'SCHW', 'BSX', 'WM', 'BDX', 'ITW', 'EL',
                'LRCX', 'APD', 'HUM', 'MMC', 'NSC', 'KLAC', 'CME', 'PLD', 'AON',
                'SHW', 'ICE', 'FIS', 'USB', 'GD', 'EMR', 'CCI', 'TGT', 'MCO'
            ]
        elif universe == "sp1500":
            # Extend with mid-cap and small-cap names for S&P 1500
            # This is a representative sample - production would use actual data
            sp500_base = self.get_universe_tickers("sp500")
            midcap_sample = [
                'EPAM', 'POOL', 'TECH', 'PAYC', 'NDAQ', 'WDC', 'MTCH', 'DLTR',
                'JBHT', 'EXPD', 'VRSN', 'SWKS', 'LW', 'ALGN', 'MKTX', 'CDAY',
                'ZBRA', 'DGX', 'CAH', 'CTXS', 'ULTA', 'HOLX', 'PKI', 'ANSS',
                'MPWR', 'ALLE', 'TPG', 'WST', 'CDW', 'ODFL', 'TDY', 'GPN',
                'ARE', 'CSGP', 'LYV', 'TRMB', 'MCHP', 'REG', 'HSY', 'WAT',
                'DFS', 'NTRS', 'FTV', 'HSIC', 'BR', 'STE', 'VRSK', 'FITB',
                'HBAN', 'RF', 'K', 'SBNY', 'CPB', 'MKC', 'LH', 'CINF'
            ]
            tickers = sp500_base + midcap_sample
        else:
            # Default to S&P 500 sample
            tickers = self.get_universe_tickers("sp500")
            
        return tickers
    
    def download_data(self, 
                     tickers: List[str] = None,
                     start_date: str = None,
                     end_date: str = None,
                     force_refresh: bool = False) -> pd.DataFrame:
        """
        Download OHLCV data for specified tickers and date range.
        
        Args:
            tickers: List of ticker symbols
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            force_refresh: Whether to force re-download existing data
            
        Returns:
            DataFrame with OHLCV data
        """
        if tickers is None:
            tickers = self.get_universe_tickers()
        if start_date is None:
            start_date = self.data_config['start_date']
        if end_date is None:
            end_date = self.data_config['end_date']
            
        cache_file = self.data_path / f"ohlcv_{len(tickers)}stocks.parquet"
        
        if cache_file.exists() and not force_refresh:
            logger.info(f"Loading cached data from {cache_file}")
            return pd.read_parquet(cache_file)
        
        logger.info(f"Downloading data for {len(tickers)} tickers from {start_date} to {end_date}")
        
        data_frames = []
        failed_tickers = []
        
        for i, ticker in enumerate(tickers):
            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(start=start_date, end=end_date, auto_adjust=True)
                
                if len(hist) > 0:
                    hist = hist.reset_index()
                    hist['ticker'] = ticker
                    data_frames.append(hist)
                    
                if (i + 1) % 50 == 0:
                    logger.info(f"Downloaded {i + 1}/{len(tickers)} tickers")
                    
            except Exception as e:
                logger.warning(f"Failed to download {ticker}: {e}")
                failed_tickers.append(ticker)
                continue
        
        if failed_tickers:
            logger.warning(f"Failed to download {len(failed_tickers)} tickers: {failed_tickers[:10]}...")
        
        if not data_frames:
            raise ValueError("No data was successfully downloaded")
            
        df = pd.concat(data_frames, ignore_index=True)
        df = df.sort_values(['ticker', 'Date']).reset_index(drop=True)
        
        # Save to cache
        df.to_parquet(cache_file)
        logger.info(f"Saved data to {cache_file}")
        
        return df


class DataProcessor:
    """
    Handles data cleaning, preprocessing, and quality checks.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize DataProcessor with configuration.
        
        Args:
            config_path: Path to configuration YAML file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the raw OHLCV data.
        
        Args:
            df: Raw OHLCV DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        logger.info("Starting data cleaning process")
        
        df = df.copy()
        initial_rows = len(df)
        
        # Remove rows with missing essential data
        essential_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        df = df.dropna(subset=essential_cols)
        
        # Remove rows with zero or negative prices
        price_cols = ['Open', 'High', 'Low', 'Close']
        for col in price_cols:
            df = df[df[col] > 0]
        
        # Remove rows with impossible price relationships
        df = df[df['High'] >= df['Low']]
        df = df[df['High'] >= df['Close']]
        df = df[df['High'] >= df['Open']]
        df = df[df['Low'] <= df['Close']]
        df = df[df['Low'] <= df['Open']]
        
        # Remove extreme price movements (likely data errors)
        df = df.sort_values(['ticker', 'Date'])
        df['price_change'] = df.groupby('ticker')['Close'].pct_change()
        df = df[abs(df['price_change']) < 0.5]  # Remove >50% daily moves
        
        # Handle stock splits and other corporate actions
        df = self._adjust_for_splits(df)
        
        # Fill missing data
        df = self._fill_missing_data(df)
        
        final_rows = len(df)
        logger.info(f"Data cleaning complete: {initial_rows} -> {final_rows} rows ({final_rows/initial_rows:.1%} retained)")
        
        return df
    
    def _adjust_for_splits(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adjust for stock splits and other corporate actions.
        """
        # yfinance auto_adjust=True should handle most adjustments
        # Additional custom logic can be added here if needed
        return df
    
    def _fill_missing_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fill missing data using appropriate methods.
        """
        df = df.sort_values(['ticker', 'Date'])
        
        # Forward fill price data within each ticker
        price_cols = ['Open', 'High', 'Low', 'Close']
        df[price_cols] = df.groupby('ticker')[price_cols].ffill()
        
        # Fill volume with 0 for missing days
        df['Volume'] = df['Volume'].fillna(0)
        
        return df
    
    def calculate_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate various return measures.
        
        Args:
            df: Cleaned OHLCV DataFrame
            
        Returns:
            DataFrame with return columns added
        """
        df = df.copy()
        df = df.sort_values(['ticker', 'Date'])
        
        # Daily returns
        df['return_1d'] = df.groupby('ticker')['Close'].pct_change()
        
        # Multi-period returns
        for period in [5, 21, 63, 126, 252]:
            df[f'return_{period}d'] = df.groupby('ticker')['Close'].pct_change(period)
        
        # Log returns
        df['log_return_1d'] = df.groupby('ticker')['Close'].transform(lambda x: np.log(x / x.shift(1)))
        
        return df
    
    def add_market_data(self, df: pd.DataFrame, market_ticker: str = "SPY") -> pd.DataFrame:
        """
        Add market benchmark data.
        
        Args:
            df: Stock data DataFrame
            market_ticker: Market benchmark ticker
            
        Returns:
            DataFrame with market data added
        """
        # Download market data
        start_date = df['Date'].min() - timedelta(days=30)
        end_date = df['Date'].max() + timedelta(days=1)
        
        market = yf.Ticker(market_ticker)
        market_data = market.history(start=start_date, end=end_date, auto_adjust=True)
        market_data = market_data.reset_index()
        market_data['market_return'] = market_data['Close'].pct_change()
        
        # Merge with stock data
        df = df.merge(
            market_data[['Date', 'market_return']],
            on='Date',
            how='left'
        )
        
        return df
    
    def filter_universe(self, df: pd.DataFrame, 
                       min_price: float = 5.0,
                       min_market_cap: float = 1e9,
                       min_volume: float = 1e6) -> pd.DataFrame:
        """
        Filter universe based on liquidity and size requirements.
        
        Args:
            df: Stock data DataFrame
            min_price: Minimum stock price
            min_market_cap: Minimum market capitalization
            min_volume: Minimum daily dollar volume
            
        Returns:
            Filtered DataFrame
        """
        df = df.copy()
        
        # Calculate market cap (simplified - would need shares outstanding data)
        # For now, use price * volume as a proxy for liquidity
        df['dollar_volume'] = df['Close'] * df['Volume']
        
        # Apply filters
        df = df[df['Close'] >= min_price]
        df = df[df['dollar_volume'] >= min_volume]
        
        # Remove tickers with insufficient history
        min_observations = 252  # 1 year of data
        ticker_counts = df.groupby('ticker').size()
        valid_tickers = ticker_counts[ticker_counts >= min_observations].index
        df = df[df['ticker'].isin(valid_tickers)]
        
        return df


def load_and_process_data(config_path: str = "config.yaml", 
                         force_refresh: bool = False) -> pd.DataFrame:
    """
    Convenience function to load and process data in one step.
    
    Args:
        config_path: Path to configuration file
        force_refresh: Whether to force re-download of data
        
    Returns:
        Processed DataFrame ready for feature engineering
    """
    # Load data
    loader = DataLoader(config_path)
    raw_data = loader.download_data(force_refresh=force_refresh)
    
    # Process data
    processor = DataProcessor(config_path)
    clean_data = processor.clean_data(raw_data)
    data_with_returns = processor.calculate_returns(clean_data)
    data_with_market = processor.add_market_data(data_with_returns)
    final_data = processor.filter_universe(data_with_market)
    
    logger.info(f"Final dataset: {len(final_data)} observations, {final_data['ticker'].nunique()} tickers")
    
    return final_data


if __name__ == "__main__":
    # Example usage
    data = load_and_process_data(force_refresh=False)
    print(f"Loaded data shape: {data.shape}")
    print(f"Date range: {data['Date'].min()} to {data['Date'].max()}")
    print(f"Tickers: {data['ticker'].nunique()}")
    print(data.head())
