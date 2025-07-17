"""
Test configuration for the Alpha Engine test suite.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import yaml


@pytest.fixture
def sample_config():
    """Sample configuration for testing."""
    return {
        'data': {
            'universe': 'sp500',
            'start_date': '2020-01-01',
            'end_date': '2022-01-01',
            'data_path': 'data/raw/'
        },
        'backtest': {
            'start_date': '2020-01-01', 
            'end_date': '2021-01-01',
            'initial_capital': 1000000,
            'rebalance_freq': 'M'
        },
        'models': {
            'ridge': {'alpha': 1.0, 'fit_intercept': True},
            'xgboost': {'n_estimators': 10, 'max_depth': 3, 'learning_rate': 0.1},
            'neural_network': {'hidden_layers': [16, 8], 'epochs': 5, 'batch_size': 32}
        },
        'portfolio': {
            'max_position_weight': 0.01,
            'max_sector_weight': 0.10,
            'kelly_lookback': 63,
            'kelly_fraction': 0.5,
            'signal_blend': 'equal'
        },
        'features': {
            'lookback_periods': [21, 63],
            'zscore_window': 252
        },
        'output': {
            'reports_path': 'reports/',
            'plots_path': 'plots/',
            'results_path': 'results/',
            'save_weights': True,
            'generate_tearsheet': False
        }
    }


@pytest.fixture
def sample_data():
    """Sample OHLCV data for testing."""
    dates = pd.date_range('2020-01-01', '2021-12-31', freq='D')
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    
    data = []
    for ticker in tickers:
        ticker_data = pd.DataFrame({
            'Date': dates,
            'ticker': ticker,
            'Open': 100 + np.random.randn(len(dates)).cumsum(),
            'High': 100 + np.random.randn(len(dates)).cumsum() + 2,
            'Low': 100 + np.random.randn(len(dates)).cumsum() - 2,
            'Close': 100 + np.random.randn(len(dates)).cumsum(),
            'Volume': np.random.randint(1000000, 10000000, len(dates))
        })
        
        # Ensure price relationships are valid
        ticker_data['High'] = np.maximum(ticker_data['High'], ticker_data[['Open', 'Close']].max(axis=1))
        ticker_data['Low'] = np.minimum(ticker_data['Low'], ticker_data[['Open', 'Close']].min(axis=1))
        
        # Add returns
        ticker_data['return_1d'] = ticker_data['Close'].pct_change()
        ticker_data['return_21d'] = ticker_data['Close'].pct_change(21)
        
        data.append(ticker_data)
    
    return pd.concat(data, ignore_index=True).dropna()


@pytest.fixture
def temp_config_file(sample_config):
    """Create temporary config file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(sample_config, f)
        return f.name


@pytest.fixture
def sample_features_data(sample_data):
    """Sample data with basic features for testing."""
    data = sample_data.copy()
    
    # Add some basic features
    data = data.sort_values(['ticker', 'Date'])
    
    # Momentum features
    data['momentum_12_1'] = data.groupby('ticker')['return_21d'].shift(1)
    data['short_term_reversal'] = data.groupby('ticker')['return_1d'].shift(1)
    
    # Volatility features  
    data['realized_vol_21d'] = data.groupby('ticker')['return_1d'].rolling(21).std()
    
    # Technical features
    data['rsi_14'] = data.groupby('ticker')['Close'].transform(
        lambda x: 50 + 50 * np.tanh((x - x.rolling(14).mean()) / x.rolling(14).std())
    )
    
    # Z-scored features
    feature_cols = ['momentum_12_1', 'short_term_reversal', 'realized_vol_21d', 'rsi_14']
    for col in feature_cols:
        data[f'{col}_zscore'] = data.groupby('Date')[col].transform(
            lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
        )
    
    return data.dropna()


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
