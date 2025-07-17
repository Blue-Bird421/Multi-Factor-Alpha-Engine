"""
Portfolio optimization and construction module for the Alpha Engine.

Implements Kelly sizing, risk constraints, and dollar-neutral long-short
portfolio construction with sector limits.
"""

import pandas as pd
import numpy as np
import yaml
from scipy.optimize import minimize
from typing import Dict, List, Optional, Tuple
import logging
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class KellySizer:
    """
    Kelly criterion sizing for portfolio optimization.
    """
    
    def __init__(self, lookback_window: int = 126, kelly_fraction: float = 0.5):
        """
        Initialize Kelly sizer.
        
        Args:
            lookback_window: Number of days for covariance estimation
            kelly_fraction: Fraction of Kelly to use (for risk management)
        """
        self.lookback_window = lookback_window
        self.kelly_fraction = kelly_fraction
        
    def calculate_kelly_weights(self, 
                               expected_returns: pd.Series,
                               return_history: pd.DataFrame,
                               max_weight: float = 0.05) -> pd.Series:
        """
        Calculate Kelly-optimal weights.
        
        The Kelly formula for portfolio optimization:
        w* = kelly_fraction * Σ^(-1) * μ
        
        Args:
            expected_returns: Expected returns for each asset
            return_history: Historical returns for covariance estimation
            max_weight: Maximum weight per position
            
        Returns:
            Kelly-optimal weights
        """
        # Estimate covariance matrix
        returns_window = return_history.tail(self.lookback_window)
        if len(returns_window) < 50:  # Minimum observations
            logger.warning(f"Insufficient data for covariance estimation: {len(returns_window)} observations")
            # Return equal weights
            n_assets = len(expected_returns)
            equal_weight = 1.0 / n_assets if n_assets > 0 else 0
            return pd.Series(equal_weight, index=expected_returns.index)
        
        # Calculate covariance matrix with regularization
        cov_matrix = returns_window.cov()
        
        # Add regularization (shrinkage towards diagonal matrix)
        shrinkage = 0.1
        diag_cov = np.diag(np.diag(cov_matrix))
        cov_matrix = (1 - shrinkage) * cov_matrix + shrinkage * diag_cov
        
        try:
            # Calculate Kelly weights: w = kelly_fraction * Σ^(-1) * μ
            inv_cov = np.linalg.inv(cov_matrix.values)
            kelly_weights = self.kelly_fraction * np.dot(inv_cov, expected_returns.values)
            
            # Create Series with proper index
            weights = pd.Series(kelly_weights, index=expected_returns.index)
            
            # Apply maximum weight constraint
            weights = weights.clip(-max_weight, max_weight)
            
            return weights
            
        except np.linalg.LinAlgError:
            logger.warning("Covariance matrix is singular, using equal weights")
            n_assets = len(expected_returns)
            equal_weight = 1.0 / n_assets if n_assets > 0 else 0
            return pd.Series(equal_weight, index=expected_returns.index)


class PortfolioOptimizer:
    """
    Portfolio optimizer with constraints and risk management.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize portfolio optimizer.
        
        Args:
            config_path: Path to configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        portfolio_config = self.config['portfolio']
        self.max_position_weight = portfolio_config.get('max_position_weight', 0.01)
        self.max_sector_weight = portfolio_config.get('max_sector_weight', 0.10)
        self.kelly_lookback = portfolio_config.get('kelly_lookback', 126)
        self.kelly_fraction = portfolio_config.get('kelly_fraction', 0.5)
        
        self.kelly_sizer = KellySizer(self.kelly_lookback, self.kelly_fraction)
        
    def create_sector_mapping(self, tickers: List[str]) -> Dict[str, str]:
        """
        Create sector mapping for tickers.
        
        This is a simplified mapping - in production you'd use 
        actual sector data from financial APIs.
        
        Args:
            tickers: List of ticker symbols
            
        Returns:
            Dictionary mapping ticker to sector
        """
        # Simplified sector mapping based on common ticker patterns
        sector_map = {}
        
        # Technology
        tech_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 
                       'ADBE', 'CRM', 'ORCL', 'IBM', 'INTC', 'AMD', 'QCOM']
        
        # Financial
        financial_tickers = ['JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'AXP', 'USB', 
                           'PNC', 'TFC', 'SCHW', 'BLK', 'SPGI']
        
        # Healthcare
        healthcare_tickers = ['JNJ', 'UNH', 'PFE', 'ABBV', 'TMO', 'ABT', 'MRK',
                            'DHR', 'BMY', 'LLY', 'AMGN', 'GILD', 'MDT']
        
        # Energy
        energy_tickers = ['XOM', 'CVX', 'COP', 'EOG', 'SLB', 'PSX', 'VLO', 'MPC']
        
        # Consumer
        consumer_tickers = ['PG', 'KO', 'PEP', 'WMT', 'HD', 'NKE', 'DIS', 'MCD',
                          'COST', 'TJX', 'SBUX', 'TGT']
        
        # Industrial
        industrial_tickers = ['CAT', 'HON', 'UPS', 'GE', 'MMM', 'BA', 'LMT', 'RTX',
                            'DE', 'EMR', 'ITW', 'GD']
        
        for ticker in tickers:
            if ticker in tech_tickers:
                sector_map[ticker] = 'Technology'
            elif ticker in financial_tickers:
                sector_map[ticker] = 'Financial'
            elif ticker in healthcare_tickers:
                sector_map[ticker] = 'Healthcare'
            elif ticker in energy_tickers:
                sector_map[ticker] = 'Energy'
            elif ticker in consumer_tickers:
                sector_map[ticker] = 'Consumer'
            elif ticker in industrial_tickers:
                sector_map[ticker] = 'Industrial'
            else:
                sector_map[ticker] = 'Other'
        
        return sector_map
    
    def optimize_portfolio(self,
                          expected_returns: pd.Series,
                          return_history: pd.DataFrame,
                          current_date: pd.Timestamp) -> pd.Series:
        """
        Optimize portfolio using Kelly criterion with constraints.
        
        Args:
            expected_returns: Expected returns for each asset
            return_history: Historical returns for covariance estimation
            current_date: Current rebalancing date
            
        Returns:
            Optimal portfolio weights
        """
        if len(expected_returns) == 0:
            return pd.Series(dtype=float)
        
        # Get initial Kelly weights
        kelly_weights = self.kelly_sizer.calculate_kelly_weights(
            expected_returns, return_history, self.max_position_weight
        )
        
        # Apply constraints
        constrained_weights = self._apply_constraints(kelly_weights, expected_returns.index.tolist())
        
        # Ensure dollar neutrality
        long_weights = constrained_weights[constrained_weights > 0].sum()
        short_weights = abs(constrained_weights[constrained_weights < 0].sum())
        
        if long_weights > 0 and short_weights > 0:
            # Scale to ensure dollar neutrality
            scale_factor = min(long_weights, short_weights)
            constrained_weights[constrained_weights > 0] *= scale_factor / long_weights
            constrained_weights[constrained_weights < 0] *= scale_factor / short_weights
        
        return constrained_weights
    
    def _apply_constraints(self, weights: pd.Series, tickers: List[str]) -> pd.Series:
        """
        Apply position and sector constraints to portfolio weights.
        
        Args:
            weights: Initial portfolio weights
            tickers: List of ticker symbols
            
        Returns:
            Constrained portfolio weights
        """
        # Apply position size constraints
        weights = weights.clip(-self.max_position_weight, self.max_position_weight)
        
        # Apply sector constraints
        sector_map = self.create_sector_mapping(tickers)
        
        # Calculate current sector exposures
        sector_exposures = {}
        for ticker, weight in weights.items():
            sector = sector_map.get(ticker, 'Other')
            sector_exposures[sector] = sector_exposures.get(sector, 0) + abs(weight)
        
        # Scale down sectors that exceed limits
        for sector, exposure in sector_exposures.items():
            if exposure > self.max_sector_weight:
                scale_factor = self.max_sector_weight / exposure
                sector_tickers = [t for t, s in sector_map.items() if s == sector and t in weights.index]
                for ticker in sector_tickers:
                    weights[ticker] *= scale_factor
        
        return weights
    
    def calculate_portfolio_metrics(self, 
                                  weights: pd.Series,
                                  return_history: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate portfolio risk and return metrics.
        
        Args:
            weights: Portfolio weights
            return_history: Historical returns
            
        Returns:
            Dictionary of portfolio metrics
        """
        if len(weights) == 0 or weights.sum() == 0:
            return {
                'expected_return': 0,
                'volatility': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'long_exposure': 0,
                'short_exposure': 0,
                'net_exposure': 0,
                'gross_exposure': 0
            }
        
        # Align weights with return history
        common_assets = weights.index.intersection(return_history.columns)
        if len(common_assets) == 0:
            return self.calculate_portfolio_metrics(pd.Series(dtype=float), return_history)
        
        aligned_weights = weights[common_assets]
        aligned_returns = return_history[common_assets]
        
        # Portfolio returns
        portfolio_returns = (aligned_returns * aligned_weights).sum(axis=1)
        
        # Calculate metrics
        expected_return = portfolio_returns.mean() * 252  # Annualized
        volatility = portfolio_returns.std() * np.sqrt(252)  # Annualized
        sharpe_ratio = expected_return / volatility if volatility > 0 else 0
        
        # Calculate max drawdown
        cumulative_returns = (1 + portfolio_returns).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdowns = (cumulative_returns - rolling_max) / rolling_max
        max_drawdown = drawdowns.min()
        
        # Exposure metrics
        long_exposure = aligned_weights[aligned_weights > 0].sum()
        short_exposure = abs(aligned_weights[aligned_weights < 0].sum())
        net_exposure = long_exposure - short_exposure
        gross_exposure = long_exposure + short_exposure
        
        return {
            'expected_return': expected_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'long_exposure': long_exposure,
            'short_exposure': short_exposure,
            'net_exposure': net_exposure,
            'gross_exposure': gross_exposure
        }
    
    def create_dollar_neutral_portfolio(self, 
                                      signals: pd.Series,
                                      return_history: pd.DataFrame,
                                      capital: float = 1000000) -> Tuple[pd.Series, Dict[str, float]]:
        """
        Create a dollar-neutral long-short portfolio.
        
        Args:
            signals: Model signals/expected returns
            return_history: Historical returns for risk estimation
            capital: Total capital to allocate
            
        Returns:
            Tuple of (dollar positions, portfolio metrics)
        """
        # Get optimal weights
        weights = self.optimize_portfolio(signals, return_history, pd.Timestamp.now())
        
        if len(weights) == 0:
            return pd.Series(dtype=float), {}
        
        # Convert to dollar positions
        dollar_positions = weights * capital
        
        # Calculate portfolio metrics
        metrics = self.calculate_portfolio_metrics(weights, return_history)
        
        return dollar_positions, metrics


class PortfolioRebalancer:
    """
    Handles portfolio rebalancing logic and transaction costs.
    """
    
    def __init__(self, transaction_cost: float = 0.001, min_trade_size: float = 1000):
        """
        Initialize portfolio rebalancer.
        
        Args:
            transaction_cost: Transaction cost as fraction of trade value
            min_trade_size: Minimum trade size to avoid small transactions
        """
        self.transaction_cost = transaction_cost
        self.min_trade_size = min_trade_size
    
    def calculate_trades(self, 
                        current_positions: pd.Series,
                        target_positions: pd.Series) -> pd.Series:
        """
        Calculate required trades to reach target positions.
        
        Args:
            current_positions: Current dollar positions
            target_positions: Target dollar positions
            
        Returns:
            Required trades (positive = buy, negative = sell)
        """
        # Align indices
        all_assets = current_positions.index.union(target_positions.index)
        current_aligned = current_positions.reindex(all_assets, fill_value=0)
        target_aligned = target_positions.reindex(all_assets, fill_value=0)
        
        # Calculate required trades
        trades = target_aligned - current_aligned
        
        # Filter out small trades
        trades = trades[abs(trades) >= self.min_trade_size]
        
        return trades
    
    def calculate_transaction_costs(self, trades: pd.Series) -> float:
        """
        Calculate total transaction costs for trades.
        
        Args:
            trades: Required trades
            
        Returns:
            Total transaction costs
        """
        return abs(trades).sum() * self.transaction_cost
    
    def execute_rebalance(self,
                         current_positions: pd.Series,
                         target_positions: pd.Series) -> Tuple[pd.Series, pd.Series, float]:
        """
        Execute portfolio rebalancing.
        
        Args:
            current_positions: Current dollar positions
            target_positions: Target dollar positions
            
        Returns:
            Tuple of (new positions, trades executed, transaction costs)
        """
        trades = self.calculate_trades(current_positions, target_positions)
        transaction_costs = self.calculate_transaction_costs(trades)
        
        # Execute trades
        all_assets = current_positions.index.union(target_positions.index)
        new_positions = current_positions.reindex(all_assets, fill_value=0) + trades.reindex(all_assets, fill_value=0)
        
        logger.info(f"Executed {len(trades)} trades with total cost: ${transaction_costs:,.2f}")
        
        return new_positions, trades, transaction_costs


def construct_portfolio(signals: pd.Series,
                       return_history: pd.DataFrame,
                       config_path: str = "config.yaml",
                       capital: float = None) -> Tuple[pd.Series, Dict[str, float]]:
    """
    Convenience function to construct optimized portfolio.
    
    Args:
        signals: Model signals/expected returns
        return_history: Historical returns
        config_path: Path to configuration file
        capital: Total capital (if None, uses config value)
        
    Returns:
        Tuple of (dollar positions, portfolio metrics)
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    if capital is None:
        capital = config['backtest']['initial_capital']
    
    optimizer = PortfolioOptimizer(config_path)
    return optimizer.create_dollar_neutral_portfolio(signals, return_history, capital)


if __name__ == "__main__":
    # Example usage
    from .data import load_and_process_data
    from .features import FeatureEngine
    from .models import ModelEnsemble
    
    # Load data and train models (simplified for example)
    data = load_and_process_data()
    engine = FeatureEngine()
    featured_data = engine.engineer_features(data)
    
    # Get some sample signals (in practice, these come from trained models)
    sample_tickers = featured_data['ticker'].unique()[:50]
    sample_signals = pd.Series(np.random.normal(0, 0.02, len(sample_tickers)), 
                              index=sample_tickers)
    
    # Create sample return history
    sample_returns = featured_data[featured_data['ticker'].isin(sample_tickers)].pivot(
        index='Date', columns='ticker', values='return_1d'
    ).dropna()
    
    # Construct portfolio
    positions, metrics = construct_portfolio(sample_signals, sample_returns)
    
    print(f"Portfolio construction complete")
    print(f"Number of positions: {len(positions)}")
    print(f"Net exposure: {metrics.get('net_exposure', 0):.1%}")
    print(f"Gross exposure: {metrics.get('gross_exposure', 0):.1%}")
    print(f"Expected Sharpe ratio: {metrics.get('sharpe_ratio', 0):.2f}")
