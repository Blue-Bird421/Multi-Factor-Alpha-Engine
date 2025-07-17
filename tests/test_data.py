"""
Tests for the data module.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from alpha_engine.data import DataLoader, DataProcessor, load_and_process_data


class TestDataLoader:
    """Test DataLoader class."""
    
    def test_init(self, temp_config_file):
        """Test DataLoader initialization."""
        loader = DataLoader(temp_config_file)
        assert loader.config is not None
        assert 'data' in loader.config
        assert loader.data_path.exists()
    
    def test_get_universe_tickers_sp500(self, temp_config_file):
        """Test getting S&P 500 tickers."""
        loader = DataLoader(temp_config_file)
        tickers = loader.get_universe_tickers('sp500')
        
        assert len(tickers) > 0
        assert 'AAPL' in tickers
        assert 'MSFT' in tickers
        assert isinstance(tickers, list)
    
    def test_get_universe_tickers_sp1500(self, temp_config_file):
        """Test getting S&P 1500 tickers."""
        loader = DataLoader(temp_config_file)
        tickers = loader.get_universe_tickers('sp1500')
        
        assert len(tickers) > 100  # Should be larger than S&P 500
        assert 'AAPL' in tickers  # Should include S&P 500 stocks
    
    @patch('alpha_engine.data.yf.Ticker')
    def test_download_data_success(self, mock_ticker, temp_config_file, sample_data):
        """Test successful data download."""
        # Mock yfinance response
        mock_stock = MagicMock()
        mock_stock.history.return_value = sample_data[sample_data['ticker'] == 'AAPL'].set_index('Date')[
            ['Open', 'High', 'Low', 'Close', 'Volume']
        ]
        mock_ticker.return_value = mock_stock
        
        loader = DataLoader(temp_config_file)
        
        # Test with small ticker list
        result = loader.download_data(['AAPL'], '2020-01-01', '2020-12-31', force_refresh=True)
        
        assert isinstance(result, pd.DataFrame)
        assert 'ticker' in result.columns
        assert len(result) > 0
    
    @patch('alpha_engine.data.yf.Ticker')
    def test_download_data_failure(self, mock_ticker, temp_config_file):
        """Test data download with failures."""
        # Mock yfinance to raise exception
        mock_ticker.side_effect = Exception("Download failed")
        
        loader = DataLoader(temp_config_file)
        
        with pytest.raises(ValueError, match="No data was successfully downloaded"):
            loader.download_data(['INVALID'], '2020-01-01', '2020-12-31', force_refresh=True)


class TestDataProcessor:
    """Test DataProcessor class."""
    
    def test_init(self, temp_config_file):
        """Test DataProcessor initialization."""
        processor = DataProcessor(temp_config_file)
        assert processor.config is not None
    
    def test_clean_data(self, temp_config_file, sample_data):
        """Test data cleaning."""
        processor = DataProcessor(temp_config_file)
        
        # Add some problematic data
        dirty_data = sample_data.copy()
        dirty_data.loc[0, 'Close'] = np.nan  # Missing price
        dirty_data.loc[1, 'Close'] = -5  # Negative price
        dirty_data.loc[2, 'High'] = 50  # High < Low
        dirty_data.loc[2, 'Low'] = 100
        
        cleaned = processor.clean_data(dirty_data)
        
        assert len(cleaned) < len(dirty_data)  # Should remove problematic rows
        assert cleaned['Close'].isna().sum() == 0  # No missing prices
        assert (cleaned['Close'] > 0).all()  # No negative prices
        assert (cleaned['High'] >= cleaned['Low']).all()  # Valid price relationships
    
    def test_calculate_returns(self, temp_config_file, sample_data):
        """Test return calculations."""
        processor = DataProcessor(temp_config_file)
        result = processor.calculate_returns(sample_data)
        
        # Check that return columns exist
        assert 'return_1d' in result.columns
        assert 'return_21d' in result.columns
        assert 'log_return_1d' in result.columns
        
        # Check return calculations are reasonable
        returns = result['return_1d'].dropna()
        assert returns.abs().max() < 1.0  # No extreme returns in sample data
    
    def test_filter_universe(self, temp_config_file, sample_data):
        """Test universe filtering."""
        processor = DataProcessor(temp_config_file)
        
        # Add return data
        data_with_returns = processor.calculate_returns(sample_data)
        
        filtered = processor.filter_universe(
            data_with_returns, 
            min_price=10.0, 
            min_volume=1000
        )
        
        assert len(filtered) <= len(data_with_returns)
        assert (filtered['Close'] >= 10.0).all()


class TestIntegration:
    """Integration tests for data module."""
    
    @pytest.mark.slow
    def test_load_and_process_data(self, temp_config_file):
        """Test complete data loading and processing pipeline."""
        # This is a slow test as it may download real data
        with patch('alpha_engine.data.DataLoader.download_data') as mock_download:
            # Mock the download to return sample data
            sample_df = pd.DataFrame({
                'Date': pd.date_range('2020-01-01', '2020-12-31'),
                'ticker': 'AAPL',
                'Open': np.random.randn(365).cumsum() + 100,
                'High': np.random.randn(365).cumsum() + 102,
                'Low': np.random.randn(365).cumsum() + 98,
                'Close': np.random.randn(365).cumsum() + 100,
                'Volume': np.random.randint(1000000, 10000000, 365)
            })
            mock_download.return_value = sample_df
            
            result = load_and_process_data(temp_config_file, force_refresh=True)
            
            assert isinstance(result, pd.DataFrame)
            assert len(result) > 0
            assert 'return_1d' in result.columns
            assert 'market_return' in result.columns or len(result) == 0  # May be empty if market data fails
    
    def test_data_quality_checks(self, sample_data):
        """Test data quality after processing."""
        processor = DataProcessor()
        
        # Process the data
        cleaned = processor.clean_data(sample_data)
        returns_added = processor.calculate_returns(cleaned)
        
        # Quality checks
        assert not returns_added['Close'].isna().any()
        assert (returns_added['Close'] > 0).all()
        assert (returns_added['Volume'] >= 0).all()
        assert returns_added['return_1d'].abs().quantile(0.99) < 0.5  # No extreme daily returns
        
        # Check data integrity
        for ticker in returns_added['ticker'].unique():
            ticker_data = returns_added[returns_added['ticker'] == ticker].sort_values('Date')
            assert ticker_data['Date'].is_monotonic_increasing
            assert len(ticker_data['Date'].unique()) == len(ticker_data)  # No duplicate dates


if __name__ == "__main__":
    pytest.main([__file__])
