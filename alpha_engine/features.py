"""
Feature engineering module for the Alpha Engine.

This module implements 25+ academic and practitioner factors including
value, momentum, quality, size, liquidity, and technical indicators.
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path
import yaml
import warnings
warnings.filterwarnings('ignore')

# Try to import talib, provide fallback if not available
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    logging.warning("TA-Lib not available. Technical indicators will use simplified implementations.")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngine:
    """
    Main feature engineering class that calculates all factors.
    """
    
    def __init__(self, config_path: str = "config.yaml", features_path: str = "features.yaml"):
        """
        Initialize FeatureEngine with configuration.
        
        Args:
            config_path: Path to main configuration file
            features_path: Path to features configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        with open(features_path, 'r') as f:
            self.features_config = yaml.safe_load(f)
            
        self.feature_config = self.config['features']
        self.lookback_periods = self.feature_config['lookback_periods']
        self.zscore_window = self.feature_config['zscore_window']
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer all features for the dataset.
        
        Args:
            df: Input DataFrame with OHLCV and return data
            
        Returns:
            DataFrame with all engineered features
        """
        logger.info("Starting feature engineering process")
        
        df = df.copy()
        df = df.sort_values(['ticker', 'Date']).reset_index(drop=True)
        
        # Calculate base features
        df = self._calculate_value_factors(df)
        df = self._calculate_momentum_factors(df)
        df = self._calculate_size_factors(df)
        df = self._calculate_quality_factors(df)
        df = self._calculate_liquidity_factors(df)
        df = self._calculate_volatility_factors(df)
        df = self._calculate_technical_factors(df)
        df = self._calculate_risk_factors(df)
        
        # Apply transformations
        df = self._apply_transformations(df)
        
        # Get feature columns
        feature_cols = self._get_feature_columns(df)
        logger.info(f"Generated {len(feature_cols)} features")
        
        return df
    
    def _calculate_value_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate value factors."""
        logger.info("Calculating value factors")
        
        # Simplified value factors (would need fundamental data for full implementation)
        # Using price-based proxies
        
        # Earnings yield proxy (inverse of PE ratio approximation)
        df['earnings_yield'] = 1 / (df['Close'] / df['return_252d'].rolling(4).mean().replace(0, np.nan))
        
        # Sales to price (using volume as proxy for business activity)
        df['sales_to_price'] = df.groupby('ticker')['Volume'].transform(lambda x: x.rolling(252).mean()) / df['Close']
        
        # Book to market (approximation using historical returns)
        df['book_to_market'] = 1 / df.groupby('ticker')['Close'].transform(lambda x: x / x.rolling(1260).mean())
        
        # Dividend yield approximation
        df['dividend_yield'] = np.maximum(0, -df.groupby('ticker')['return_1d'].transform(lambda x: x.rolling(252).mean()))
        
        return df
    
    def _calculate_momentum_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate momentum factors."""
        logger.info("Calculating momentum factors")
        
        # 12-1 month momentum (252-21 days)
        df['momentum_12_1'] = df['return_252d'] / (1 + df['return_21d']) - 1
        
        # 6-1 month momentum
        df['momentum_6_1'] = df['return_126d'] / (1 + df['return_21d']) - 1
        
        # Short-term reversal (previous month return)
        df['short_term_reversal'] = df['return_21d']
        
        # Price trend (linear regression slope)
        def calculate_price_trend(prices, window=63):
            """Calculate linear regression slope of prices."""
            trends = []
            for i in range(len(prices)):
                start_idx = max(0, i - window + 1)
                price_window = prices[start_idx:i+1]
                if len(price_window) >= 10:  # Minimum observations
                    x = np.arange(len(price_window))
                    slope = stats.linregress(x, price_window)[0]
                    trends.append(slope)
                else:
                    trends.append(np.nan)
            return trends
        
        df['price_trend'] = df.groupby('ticker')['Close'].transform(
            lambda x: pd.Series(calculate_price_trend(x.values), index=x.index)
        )
        
        # Relative strength
        if TALIB_AVAILABLE:
            df['rsi_14'] = df.groupby('ticker')['Close'].transform(
                lambda x: pd.Series(talib.RSI(x.values, timeperiod=14), index=x.index)
            )
        else:
            # Fallback RSI calculation
            def calculate_rsi(prices, window=14):
                """Calculate RSI without TA-Lib."""
                delta = prices.diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))
                return rsi
            
            df['rsi_14'] = df.groupby('ticker')['Close'].transform(calculate_rsi)
        
        return df
    
    def _calculate_size_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate size factors."""
        logger.info("Calculating size factors")
        
        # Market cap proxy (price * volume as liquidity measure)
        df['market_cap_proxy'] = df['Close'] * df['Volume']
        df['log_market_cap'] = np.log(df['market_cap_proxy'] + 1)
        
        # Relative size within universe
        df['relative_size'] = df.groupby('Date')['market_cap_proxy'].rank(pct=True)
        
        return df
    
    def _calculate_quality_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate quality factors (using return-based proxies)."""
        logger.info("Calculating quality factors")
        
        # ROE proxy (using return consistency)
        mean_returns = df.groupby('ticker')['return_1d'].transform(lambda x: x.rolling(252).mean())
        std_returns = df.groupby('ticker')['return_1d'].transform(lambda x: x.rolling(252).std())
        df['roe_proxy'] = mean_returns / std_returns
        
        # Profit margin proxy (return stability)
        df['profit_margin_proxy'] = -df.groupby('ticker')['return_1d'].transform(lambda x: x.rolling(63).std())
        
        # Accruals proxy (difference between returns and volume changes)
        volume_change = df.groupby('ticker')['Volume'].pct_change()
        df['accruals_proxy'] = df['return_1d'] - volume_change.rolling(21).mean()
        
        # Earnings quality (return predictability)
        def calculate_earnings_quality(returns):
            """Calculate earnings quality as return predictability."""
            if len(returns) < 60:
                return np.nan
            # R-squared from AR(1) model
            returns_lagged = returns.shift(1).dropna()
            returns_current = returns[1:len(returns_lagged)+1]
            if len(returns_current) > 20:
                correlation = returns_current.corr(returns_lagged)
                return correlation ** 2 if not np.isnan(correlation) else 0
            return 0
        
        df['earnings_quality'] = df.groupby('ticker')['return_1d'].rolling(126).apply(
            calculate_earnings_quality, raw=False
        )
        
        return df
    
    def _calculate_liquidity_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate liquidity factors."""
        logger.info("Calculating liquidity factors")
        
        # Turnover (volume relative to float, using moving average as proxy)
        avg_volume = df.groupby('ticker')['Volume'].rolling(252).mean()
        df['turnover'] = df['Volume'] / avg_volume
        
        # Amihud illiquidity measure
        dollar_volume = df['Close'] * df['Volume']
        df['amihud_illiquidity'] = df.groupby('ticker').apply(
            lambda x: (abs(x['return_1d']) / (dollar_volume[x.index] + 1e-10)).rolling(21).mean()
        ).values
        
        # Bid-ask spread proxy (using high-low spread)
        df['bid_ask_spread'] = (df['High'] - df['Low']) / ((df['High'] + df['Low']) / 2)
        
        # Volume trend
        df['volume_trend'] = df.groupby('ticker')['Volume'].pct_change(21)
        
        return df
    
    def _calculate_volatility_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate volatility factors."""
        logger.info("Calculating volatility factors")
        
        # Realized volatility at different horizons
        for period in [21, 63, 126]:
            df[f'realized_vol_{period}d'] = df.groupby('ticker')['return_1d'].rolling(period).std() * np.sqrt(252)
        
        # Volatility of volatility
        vol_21d = df.groupby('ticker')['return_1d'].rolling(21).std()
        df['vol_of_vol'] = df.groupby('ticker').apply(
            lambda x: vol_21d[x.index].rolling(63).std()
        ).values
        
        # Downside volatility
        negative_returns = df['return_1d'].where(df['return_1d'] < 0, 0)
        df['downside_vol'] = df.groupby('ticker')[negative_returns.name].rolling(63).std() * np.sqrt(252)
        
        # GARCH-like volatility (simple EWMA)
        df['ewma_vol'] = df.groupby('ticker')['return_1d'].ewm(span=21).std() * np.sqrt(252)
        
        return df
    
    def _calculate_technical_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical factors."""
        logger.info("Calculating technical factors")
        
        # MACD
        if TALIB_AVAILABLE:
            macd_values = df.groupby('ticker')['Close'].apply(
                lambda x: pd.Series(talib.MACD(x.values)[2], index=x.index)  # Signal line
            )
            df['macd_signal'] = macd_values.values
        else:
            # Fallback MACD calculation
            def calculate_macd_signal(prices, fast=12, slow=26, signal=9):
                """Calculate MACD signal without TA-Lib."""
                ema_fast = prices.ewm(span=fast).mean()
                ema_slow = prices.ewm(span=slow).mean()
                macd_line = ema_fast - ema_slow
                macd_signal = macd_line.ewm(span=signal).mean()
                return macd_signal
            
            df['macd_signal'] = df.groupby('ticker')['Close'].apply(calculate_macd_signal)
        
        # Bollinger Bands position
        def calculate_bollinger_position(prices, window=20, num_std=2):
            """Calculate position within Bollinger Bands."""
            sma = prices.rolling(window).mean()
            std = prices.rolling(window).std()
            upper_band = sma + (std * num_std)
            lower_band = sma - (std * num_std)
            position = (prices - lower_band) / (upper_band - lower_band)
            return position.clip(0, 1)  # Clip to [0, 1] range
        
        df['bollinger_position'] = df.groupby('ticker')['Close'].apply(calculate_bollinger_position)
        
        # Price relative to moving averages
        for period in [20, 50, 200]:
            sma = df.groupby('ticker')['Close'].rolling(period).mean()
            df[f'price_to_sma_{period}'] = df['Close'] / sma
        
        # Williams %R
        def calculate_williams_r(high, low, close, period=14):
            """Calculate Williams %R."""
            highest_high = high.rolling(period).max()
            lowest_low = low.rolling(period).min()
            wr = -100 * (highest_high - close) / (highest_high - lowest_low)
            return wr
        
        df['williams_r'] = df.groupby('ticker').apply(
            lambda x: calculate_williams_r(x['High'], x['Low'], x['Close'])
        ).values
        
        return df
    
    def _calculate_risk_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate risk factors."""
        logger.info("Calculating risk factors")
        
        # Market beta
        def calculate_beta(stock_returns, market_returns, window=252):
            """Calculate rolling market beta."""
            betas = []
            for i in range(len(stock_returns)):
                start_idx = max(0, i - window + 1)
                stock_window = stock_returns[start_idx:i+1]
                market_window = market_returns[start_idx:i+1]
                
                if len(stock_window) >= 60:  # Minimum observations
                    # Remove NaN values
                    valid_idx = ~(np.isnan(stock_window) | np.isnan(market_window))
                    if valid_idx.sum() >= 60:
                        stock_clean = stock_window[valid_idx]
                        market_clean = market_window[valid_idx]
                        
                        if np.var(market_clean) > 0:
                            beta = np.cov(stock_clean, market_clean)[0, 1] / np.var(market_clean)
                        else:
                            beta = 1.0
                    else:
                        beta = 1.0
                else:
                    beta = 1.0
                    
                betas.append(beta)
            
            return np.array(betas)
        
        if 'market_return' in df.columns:
            df['beta'] = df.groupby('ticker').apply(
                lambda x: pd.Series(
                    calculate_beta(x['return_1d'].values, x['market_return'].values),
                    index=x.index
                )
            ).values
            
            # Idiosyncratic volatility
            predicted_returns = df['beta'] * df['market_return']
            residual_returns = df['return_1d'] - predicted_returns
            df['idiosyncratic_vol'] = df.groupby('ticker').apply(
                lambda x: residual_returns[x.index].rolling(126).std() * np.sqrt(252)
            ).values
        else:
            df['beta'] = 1.0
            df['idiosyncratic_vol'] = df['realized_vol_126d']
        
        # Skewness and kurtosis
        df['return_skewness'] = df.groupby('ticker')['return_1d'].rolling(126).skew()
        df['return_kurtosis'] = df.groupby('ticker')['return_1d'].rolling(126).apply(
            lambda x: stats.kurtosis(x.dropna())
        )
        
        return df
    
    def _apply_transformations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply transformations like z-scoring and winsorization."""
        logger.info("Applying feature transformations")
        
        feature_cols = self._get_feature_columns(df)
        
        for col in feature_cols:
            if col in df.columns:
                # Winsorize at 1st and 99th percentiles
                df[col] = df.groupby('Date')[col].transform(
                    lambda x: np.clip(x, x.quantile(0.01), x.quantile(0.99))
                )
                
                # Cross-sectional z-score (rank-based)
                df[f"{col}_zscore"] = df.groupby('Date')[col].transform(
                    lambda x: (x.rank() - x.rank().mean()) / x.rank().std()
                )
        
        return df
    
    def _get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """Get list of feature columns (excluding basic OHLCV and metadata)."""
        exclude_cols = [
            'Date', 'ticker', 'Open', 'High', 'Low', 'Close', 'Volume',
            'return_1d', 'return_5d', 'return_21d', 'return_63d', 'return_126d', 'return_252d',
            'log_return_1d', 'market_return', 'price_change', 'dollar_volume'
        ]
        
        feature_cols = [col for col in df.columns if col not in exclude_cols and not col.endswith('_zscore')]
        return feature_cols
    
    def get_feature_matrix(self, df: pd.DataFrame, 
                          use_zscore: bool = True,
                          start_date: str = None,
                          end_date: str = None) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Get feature matrix and target variable for modeling.
        
        Args:
            df: DataFrame with engineered features
            use_zscore: Whether to use z-scored features
            start_date: Start date for filtering
            end_date: End date for filtering
            
        Returns:
            Tuple of (features DataFrame, target Series)
        """
        df_filtered = df.copy()
        
        # Filter by date range
        if start_date:
            df_filtered = df_filtered[df_filtered['Date'] >= start_date]
        if end_date:
            df_filtered = df_filtered[df_filtered['Date'] <= end_date]
        
        # Get feature columns
        if use_zscore:
            feature_cols = [col for col in df_filtered.columns if col.endswith('_zscore')]
        else:
            feature_cols = self._get_feature_columns(df_filtered)
        
        # Remove rows with missing features
        feature_data = df_filtered[['Date', 'ticker'] + feature_cols].dropna()
        
        # Target variable (next month forward return)
        # Shift returns forward to predict future performance
        target_data = df_filtered[['Date', 'ticker', 'return_21d']].copy()
        target_data['Date'] = target_data['Date'] + pd.DateOffset(months=1)
        target_data = target_data.rename(columns={'return_21d': 'target_return'})
        
        # Merge features with forward returns
        merged_data = feature_data.merge(
            target_data, on=['Date', 'ticker'], how='left'
        ).dropna()
        
        if len(merged_data) == 0:
            raise ValueError("No valid feature-target pairs found")
        
        X = merged_data[feature_cols]
        y = merged_data['target_return']
        
        logger.info(f"Feature matrix shape: {X.shape}, Target length: {len(y)}")
        
        return X, y


def engineer_all_features(df: pd.DataFrame, 
                         config_path: str = "config.yaml",
                         features_path: str = "features.yaml") -> pd.DataFrame:
    """
    Convenience function to engineer all features.
    
    Args:
        df: Input DataFrame with OHLCV data
        config_path: Path to configuration file
        features_path: Path to features configuration file
        
    Returns:
        DataFrame with all engineered features
    """
    engine = FeatureEngine(config_path, features_path)
    return engine.engineer_features(df)


if __name__ == "__main__":
    # Example usage
    from .data import load_and_process_data
    
    # Load data
    data = load_and_process_data()
    
    # Engineer features
    engine = FeatureEngine()
    featured_data = engine.engineer_features(data)
    
    print(f"Original columns: {len(data.columns)}")
    print(f"With features: {len(featured_data.columns)}")
    
    # Get feature matrix
    X, y = engine.get_feature_matrix(featured_data)
    print(f"Feature matrix: {X.shape}")
    print(f"Target vector: {y.shape}")
