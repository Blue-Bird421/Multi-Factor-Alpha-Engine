"""
Backtesting and performance analysis module for the Alpha Engine.

Implements walk-forward backtesting, performance metrics calculation,
and visualization of results.
"""

import pandas as pd
import numpy as np
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
import quantstats as qs
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style for plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class Backtester:
    """
    Walk-forward backtesting engine for the alpha strategy.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize backtester with configuration.
        
        Args:
            config_path: Path to configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.backtest_config = self.config['backtest']
        self.start_date = pd.to_datetime(self.backtest_config['start_date'])
        self.end_date = pd.to_datetime(self.backtest_config['end_date'])
        self.initial_capital = self.backtest_config['initial_capital']
        self.rebalance_freq = self.backtest_config['rebalance_freq']
        
        # Initialize tracking variables
        self.results = {}
        self.positions_history = []
        self.returns_history = []
        self.metrics_history = []
        
    def run_backtest(self,
                    featured_data: pd.DataFrame,
                    model_ensemble: Any,
                    portfolio_optimizer: Any) -> Dict[str, Any]:
        """
        Run walk-forward backtest.
        
        Args:
            featured_data: DataFrame with engineered features
            model_ensemble: Trained model ensemble
            portfolio_optimizer: Portfolio optimizer instance
            
        Returns:
            Dictionary containing backtest results
        """
        logger.info(f"Starting backtest from {self.start_date} to {self.end_date}")
        
        # Filter data to backtest period
        backtest_data = featured_data[
            (featured_data['Date'] >= self.start_date) & 
            (featured_data['Date'] <= self.end_date)
        ].copy()
        
        # Get rebalancing dates
        rebalance_dates = self._get_rebalance_dates(backtest_data)
        
        # Initialize portfolio tracking
        current_capital = self.initial_capital
        current_positions = pd.Series(dtype=float)
        
        # Walk-forward backtesting
        for i, rebal_date in enumerate(rebalance_dates[:-1]):
            next_date = rebalance_dates[i + 1]
            
            logger.info(f"Rebalancing on {rebal_date.date()}")
            
            try:
                # Get features for prediction
                features_data = self._prepare_features_for_date(backtest_data, rebal_date)
                
                if len(features_data) == 0:
                    logger.warning(f"No features available for {rebal_date}")
                    continue
                
                # Generate signals
                signals = self._generate_signals(features_data, model_ensemble)
                
                if len(signals) == 0:
                    logger.warning(f"No signals generated for {rebal_date}")
                    continue
                
                # Get return history for risk estimation
                return_history = self._get_return_history(backtest_data, rebal_date)
                
                # Construct portfolio
                target_positions, portfolio_metrics = portfolio_optimizer.create_dollar_neutral_portfolio(
                    signals, return_history, current_capital
                )
                
                # Calculate period returns
                period_returns = self._calculate_period_returns(
                    current_positions, target_positions, backtest_data, rebal_date, next_date
                )
                
                # Update capital and positions
                current_capital *= (1 + period_returns)
                current_positions = target_positions
                
                # Store results
                self._store_period_results(rebal_date, current_positions, period_returns, portfolio_metrics)
                
            except Exception as e:
                logger.error(f"Error during rebalancing on {rebal_date}: {e}")
                continue
        
        # Compile final results
        self.results = self._compile_results()
        
        logger.info("Backtest completed successfully")
        return self.results
    
    def _get_rebalance_dates(self, data: pd.DataFrame) -> List[pd.Timestamp]:
        """Get list of rebalancing dates based on frequency."""
        all_dates = sorted(data['Date'].unique())
        
        if self.rebalance_freq == 'M':  # Monthly
            # Get first trading day of each month
            rebalance_dates = []
            current_month = None
            
            for date in all_dates:
                if current_month != date.month:
                    rebalance_dates.append(date)
                    current_month = date.month
                    
        elif self.rebalance_freq == 'Q':  # Quarterly
            rebalance_dates = []
            current_quarter = None
            
            for date in all_dates:
                quarter = (date.month - 1) // 3 + 1
                if current_quarter != quarter:
                    rebalance_dates.append(date)
                    current_quarter = quarter
                    
        else:  # Default to monthly
            rebalance_dates = self._get_rebalance_dates(data.assign(rebalance_freq='M'))
        
        return [d for d in rebalance_dates if self.start_date <= d <= self.end_date]
    
    def _prepare_features_for_date(self, data: pd.DataFrame, date: pd.Timestamp) -> pd.DataFrame:
        """Prepare feature data for a specific rebalancing date."""
        # Get most recent features before the rebalancing date
        available_data = data[data['Date'] <= date]
        
        # Get the latest features for each ticker
        latest_features = available_data.groupby('ticker').last().reset_index()
        
        # Filter out stale data (more than 30 days old)
        max_staleness = timedelta(days=30)
        latest_features = latest_features[
            (date - latest_features['Date']) <= max_staleness
        ]
        
        return latest_features
    
    def _generate_signals(self, features_data: pd.DataFrame, model_ensemble: Any) -> pd.Series:
        """Generate trading signals using the model ensemble."""
        if not hasattr(model_ensemble, 'predict'):
            logger.error("Model ensemble does not have predict method")
            return pd.Series(dtype=float)
        
        # Get feature columns (z-scored features)
        feature_cols = [col for col in features_data.columns if col.endswith('_zscore')]
        
        if len(feature_cols) == 0:
            logger.warning("No z-scored features found")
            return pd.Series(dtype=float)
        
        # Prepare feature matrix
        X = features_data[feature_cols].fillna(0)
        
        if len(X) == 0:
            return pd.Series(dtype=float)
        
        try:
            # Generate predictions
            predictions = model_ensemble.predict(X)
            
            # Convert to Series with ticker index
            signals = pd.Series(predictions, index=features_data['ticker'])
            
            # Remove extreme outliers
            signals = signals.clip(signals.quantile(0.01), signals.quantile(0.99))
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating signals: {e}")
            return pd.Series(dtype=float)
    
    def _get_return_history(self, data: pd.DataFrame, current_date: pd.Timestamp, 
                           lookback_days: int = 252) -> pd.DataFrame:
        """Get historical returns for risk estimation."""
        start_date = current_date - timedelta(days=lookback_days)
        
        # Filter data
        historical_data = data[
            (data['Date'] >= start_date) & 
            (data['Date'] <= current_date)
        ]
        
        # Pivot to get returns matrix
        returns_matrix = historical_data.pivot(
            index='Date', columns='ticker', values='return_1d'
        )
        
        return returns_matrix.dropna(axis=1, thresh=len(returns_matrix)*0.8)  # At least 80% data
    
    def _calculate_period_returns(self,
                                 current_positions: pd.Series,
                                 target_positions: pd.Series,
                                 data: pd.DataFrame,
                                 start_date: pd.Timestamp,
                                 end_date: pd.Timestamp) -> float:
        """Calculate portfolio returns for the period."""
        # Get actual returns for the period
        period_data = data[
            (data['Date'] > start_date) & 
            (data['Date'] <= end_date)
        ]
        
        if len(period_data) == 0:
            return 0.0
        
        # Calculate average return for each ticker during the period
        ticker_returns = period_data.groupby('ticker')['return_1d'].mean()
        
        # Use target positions for return calculation (assuming immediate execution)
        if len(target_positions) == 0:
            return 0.0
        
        # Calculate portfolio return
        common_tickers = target_positions.index.intersection(ticker_returns.index)
        if len(common_tickers) == 0:
            return 0.0
        
        position_weights = target_positions[common_tickers] / target_positions[common_tickers].abs().sum()
        period_return = (position_weights * ticker_returns[common_tickers]).sum()
        
        return period_return
    
    def _store_period_results(self,
                             date: pd.Timestamp,
                             positions: pd.Series,
                             returns: float,
                             metrics: Dict[str, float]) -> None:
        """Store results for the period."""
        self.positions_history.append({
            'date': date,
            'positions': positions.copy(),
            'capital': self.initial_capital * (1 + sum(self.returns_history))
        })
        
        self.returns_history.append(returns)
        
        self.metrics_history.append({
            'date': date,
            'return': returns,
            **metrics
        })
    
    def _compile_results(self) -> Dict[str, Any]:
        """Compile final backtest results."""
        if not self.returns_history:
            return {'error': 'No backtest results available'}
        
        # Convert to DataFrame
        returns_df = pd.DataFrame({
            'date': [entry['date'] for entry in self.positions_history],
            'return': self.returns_history
        })
        returns_df = returns_df.set_index('date')
        
        # Calculate cumulative returns
        returns_df['cumulative_return'] = (1 + returns_df['return']).cumprod()
        returns_df['portfolio_value'] = self.initial_capital * returns_df['cumulative_return']
        
        # Calculate performance metrics
        total_return = returns_df['cumulative_return'].iloc[-1] - 1
        annualized_return = (1 + total_return) ** (252 / len(returns_df)) - 1
        
        annual_vol = returns_df['return'].std() * np.sqrt(252)
        sharpe_ratio = annualized_return / annual_vol if annual_vol > 0 else 0
        
        # Calculate max drawdown
        rolling_max = returns_df['cumulative_return'].expanding().max()
        drawdowns = (returns_df['cumulative_return'] - rolling_max) / rolling_max
        max_drawdown = drawdowns.min()
        
        # Hit rate
        hit_rate = (returns_df['return'] > 0).mean()
        
        # Compile results
        results = {
            'returns_series': returns_df,
            'metrics': {
                'total_return': total_return,
                'annualized_return': annualized_return,
                'annual_volatility': annual_vol,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'hit_rate': hit_rate,
                'total_periods': len(returns_df),
                'start_date': self.start_date,
                'end_date': self.end_date,
                'initial_capital': self.initial_capital,
                'final_capital': returns_df['portfolio_value'].iloc[-1]
            },
            'positions_history': self.positions_history,
            'metrics_history': self.metrics_history
        }
        
        return results


class PerformanceAnalyzer:
    """
    Performance analysis and visualization for backtest results.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize performance analyzer.
        
        Args:
            config_path: Path to configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Create output directories
        self.plots_dir = Path(self.config['output']['plots_path'])
        self.reports_dir = Path(self.config['output']['reports_path'])
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
    
    def analyze_performance(self, backtest_results: Dict[str, Any], 
                          benchmark_data: pd.DataFrame = None) -> Dict[str, Any]:
        """
        Comprehensive performance analysis.
        
        Args:
            backtest_results: Results from backtester
            benchmark_data: Benchmark returns data
            
        Returns:
            Dictionary with analysis results
        """
        logger.info("Starting performance analysis")
        
        if 'error' in backtest_results:
            logger.error(f"Cannot analyze results: {backtest_results['error']}")
            return {}
        
        returns_df = backtest_results['returns_series']
        metrics = backtest_results['metrics']
        
        # Generate plots
        self._plot_equity_curve(returns_df)
        self._plot_drawdown_chart(returns_df)
        self._plot_rolling_metrics(returns_df)
        self._plot_monthly_heatmap(returns_df)
        
        # Compare to benchmark if available
        if benchmark_data is not None:
            benchmark_analysis = self._analyze_vs_benchmark(returns_df, benchmark_data)
            metrics.update(benchmark_analysis)
        
        # Generate QuantStats report
        if self.config['output'].get('generate_tearsheet', True):
            self._generate_quantstats_report(returns_df)
        
        # Create summary report
        self._create_summary_report(metrics)
        
        logger.info("Performance analysis completed")
        
        return {
            'metrics': metrics,
            'plots_generated': True,
            'reports_generated': True
        }
    
    def _plot_equity_curve(self, returns_df: pd.DataFrame) -> None:
        """Plot equity curve."""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.plot(returns_df.index, returns_df['portfolio_value'], 
               linewidth=2, label='Alpha Strategy')
        
        ax.set_title('Portfolio Equity Curve', fontsize=16, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Portfolio Value ($)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Format y-axis as currency
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'equity_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_drawdown_chart(self, returns_df: pd.DataFrame) -> None:
        """Plot drawdown chart."""
        # Calculate drawdowns
        rolling_max = returns_df['cumulative_return'].expanding().max()
        drawdowns = (returns_df['cumulative_return'] - rolling_max) / rolling_max
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.fill_between(returns_df.index, drawdowns, 0, alpha=0.3, color='red')
        ax.plot(returns_df.index, drawdowns, color='red', linewidth=1)
        
        ax.set_title('Portfolio Drawdowns', fontsize=16, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Drawdown (%)')
        ax.grid(True, alpha=0.3)
        
        # Format y-axis as percentage
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1%}'))
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'drawdown_chart.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_rolling_metrics(self, returns_df: pd.DataFrame) -> None:
        """Plot rolling performance metrics."""
        # Calculate rolling metrics
        window = 252  # 1 year
        rolling_return = returns_df['return'].rolling(window).mean() * 252
        rolling_vol = returns_df['return'].rolling(window).std() * np.sqrt(252)
        rolling_sharpe = rolling_return / rolling_vol
        
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        # Rolling return
        axes[0].plot(returns_df.index, rolling_return)
        axes[0].set_title('Rolling 1-Year Annualized Return')
        axes[0].set_ylabel('Return (%)')
        axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1%}'))
        axes[0].grid(True, alpha=0.3)
        
        # Rolling volatility
        axes[1].plot(returns_df.index, rolling_vol, color='orange')
        axes[1].set_title('Rolling 1-Year Annualized Volatility')
        axes[1].set_ylabel('Volatility (%)')
        axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1%}'))
        axes[1].grid(True, alpha=0.3)
        
        # Rolling Sharpe ratio
        axes[2].plot(returns_df.index, rolling_sharpe, color='green')
        axes[2].set_title('Rolling 1-Year Sharpe Ratio')
        axes[2].set_xlabel('Date')
        axes[2].set_ylabel('Sharpe Ratio')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'rolling_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_monthly_heatmap(self, returns_df: pd.DataFrame) -> None:
        """Plot monthly returns heatmap."""
        # Convert to monthly returns
        monthly_returns = returns_df['return'].groupby([
            returns_df.index.year, 
            returns_df.index.month
        ]).apply(lambda x: (1 + x).prod() - 1)
        
        # Reshape for heatmap
        monthly_pivot = monthly_returns.unstack(level=1)
        monthly_pivot.columns = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        sns.heatmap(monthly_pivot, annot=True, fmt='.1%', cmap='RdYlGn', 
                   center=0, ax=ax, cbar_kws={'label': 'Monthly Return'})
        
        ax.set_title('Monthly Returns Heatmap', fontsize=16, fontweight='bold')
        ax.set_xlabel('Month')
        ax.set_ylabel('Year')
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'monthly_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _analyze_vs_benchmark(self, returns_df: pd.DataFrame, 
                            benchmark_data: pd.DataFrame) -> Dict[str, float]:
        """Analyze performance versus benchmark."""
        # Align dates
        common_dates = returns_df.index.intersection(benchmark_data.index)
        if len(common_dates) == 0:
            return {}
        
        portfolio_returns = returns_df.loc[common_dates, 'return']
        benchmark_returns = benchmark_data.loc[common_dates, 'return']
        
        # Calculate excess returns
        excess_returns = portfolio_returns - benchmark_returns
        
        # Calculate metrics
        tracking_error = excess_returns.std() * np.sqrt(252)
        information_ratio = excess_returns.mean() * 252 / tracking_error if tracking_error > 0 else 0
        
        # Beta calculation
        beta = np.cov(portfolio_returns, benchmark_returns)[0, 1] / np.var(benchmark_returns)
        
        return {
            'tracking_error': tracking_error,
            'information_ratio': information_ratio,
            'beta': beta,
            'excess_return': excess_returns.mean() * 252
        }
    
    def _generate_quantstats_report(self, returns_df: pd.DataFrame) -> None:
        """Generate QuantStats HTML tearsheet."""
        try:
            qs.reports.html(
                returns_df['return'],
                output=str(self.reports_dir / 'tearsheet.html'),
                title='Multi-Factor Alpha Engine Performance',
                download_filename=str(self.reports_dir / 'tearsheet.html')
            )
            logger.info("QuantStats tearsheet generated successfully")
        except Exception as e:
            logger.error(f"Failed to generate QuantStats report: {e}")
    
    def _create_summary_report(self, metrics: Dict[str, float]) -> None:
        """Create summary performance report."""
        report_lines = [
            "Multi-Factor Equity Alpha Engine - Performance Summary",
            "=" * 55,
            "",
            f"Backtest Period: {metrics['start_date'].date()} to {metrics['end_date'].date()}",
            f"Initial Capital: ${metrics['initial_capital']:,.0f}",
            f"Final Capital: ${metrics['final_capital']:,.0f}",
            "",
            "Performance Metrics:",
            "-" * 20,
            f"Total Return: {metrics['total_return']:.1%}",
            f"Annualized Return: {metrics['annualized_return']:.1%}",
            f"Annual Volatility: {metrics['annual_volatility']:.1%}",
            f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}",
            f"Maximum Drawdown: {metrics['max_drawdown']:.1%}",
            f"Hit Rate: {metrics['hit_rate']:.1%}",
            "",
            f"Total Rebalancing Periods: {metrics['total_periods']}",
            "",
            "Generated by Multi-Factor Alpha Engine",
            f"Report Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        ]
        
        # Add benchmark comparison if available
        if 'information_ratio' in metrics:
            report_lines.extend([
                "",
                "Benchmark Comparison:",
                "-" * 20,
                f"Information Ratio: {metrics['information_ratio']:.2f}",
                f"Tracking Error: {metrics['tracking_error']:.1%}",
                f"Beta: {metrics['beta']:.2f}",
                f"Excess Return: {metrics['excess_return']:.1%}"
            ])
        
        # Save report
        with open(self.reports_dir / 'performance_summary.txt', 'w') as f:
            f.write('\n'.join(report_lines))
        
        logger.info("Summary report created successfully")


def run_full_backtest(featured_data: pd.DataFrame,
                     model_ensemble: Any,
                     portfolio_optimizer: Any,
                     config_path: str = "config.yaml",
                     benchmark_ticker: str = "SPY") -> Dict[str, Any]:
    """
    Run complete backtest with analysis.
    
    Args:
        featured_data: DataFrame with engineered features
        model_ensemble: Trained model ensemble
        portfolio_optimizer: Portfolio optimizer instance
        config_path: Path to configuration file
        benchmark_ticker: Benchmark ticker for comparison
        
    Returns:
        Complete backtest and analysis results
    """
    # Run backtest
    backtester = Backtester(config_path)
    backtest_results = backtester.run_backtest(featured_data, model_ensemble, portfolio_optimizer)
    
    # Analyze performance
    analyzer = PerformanceAnalyzer(config_path)
    
    # Load benchmark data if specified
    benchmark_data = None
    if benchmark_ticker:
        try:
            import yfinance as yf
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            start_date = config['backtest']['start_date']
            end_date = config['backtest']['end_date']
            
            benchmark = yf.Ticker(benchmark_ticker)
            benchmark_hist = benchmark.history(start=start_date, end=end_date)
            benchmark_data = pd.DataFrame({
                'return': benchmark_hist['Close'].pct_change()
            }).dropna()
            
        except Exception as e:
            logger.warning(f"Could not load benchmark data: {e}")
    
    analysis_results = analyzer.analyze_performance(backtest_results, benchmark_data)
    
    return {
        'backtest_results': backtest_results,
        'analysis_results': analysis_results
    }


if __name__ == "__main__":
    # Example usage would require full pipeline
    logger.info("Backtest module loaded successfully")
    print("Use run_full_backtest() to execute complete backtesting pipeline")
