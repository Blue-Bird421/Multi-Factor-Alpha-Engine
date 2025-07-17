#!/usr/bin/env python3
"""
Main pipeline script for the Multi-Factor Equity Alpha Engine.

This script runs the complete end-to-end pipeline:
1. Data ingestion and processing
2. Feature engineering  
3. Model training
4. Portfolio construction
5. Backtesting
6. Performance analysis and reporting

Usage:
    python run_pipeline.py --start 2005-01-01 --end 2024-06-30 --universe sp1500 --capital 1000000
"""

import argparse
import yaml
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import alpha engine modules
from alpha_engine.data import load_and_process_data
from alpha_engine.features import FeatureEngine
from alpha_engine.models import ModelEnsemble
from alpha_engine.portfolio import PortfolioOptimizer
from alpha_engine.backtest import run_full_backtest

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run the complete Multi-Factor Alpha Engine pipeline')
    
    parser.add_argument('--start', type=str, default='2005-01-01',
                       help='Backtest start date (YYYY-MM-DD)')
    
    parser.add_argument('--end', type=str, default='2024-06-30',
                       help='Backtest end date (YYYY-MM-DD)')
    
    parser.add_argument('--universe', type=str, default='sp1500',
                       choices=['sp500', 'sp1500', 'russell3000'],
                       help='Stock universe to use')
    
    parser.add_argument('--capital', type=float, default=1000000,
                       help='Initial capital for backtesting')
    
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file')
    
    parser.add_argument('--force-refresh', action='store_true',
                       help='Force refresh of all data and models')
    
    parser.add_argument('--skip-training', action='store_true',
                       help='Skip model training and use existing models')
    
    parser.add_argument('--benchmark', type=str, default='SPY',
                       help='Benchmark ticker for comparison')
    
    return parser.parse_args()


def update_config(args, config_path: str):
    """Update configuration with command line arguments."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Update config with arguments
    config['data']['universe'] = args.universe
    config['backtest']['start_date'] = args.start
    config['backtest']['end_date'] = args.end
    config['backtest']['initial_capital'] = args.capital
    
    # Save updated config
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    logger.info(f"Configuration updated with command line arguments")
    return config


def load_or_prepare_data(config_path: str, force_refresh: bool = False) -> pd.DataFrame:
    """Load or prepare featured dataset."""
    logger.info("=" * 60)
    logger.info("STEP 1: DATA LOADING AND FEATURE ENGINEERING")
    logger.info("=" * 60)
    
    # Check if cached featured data exists
    cache_file = Path("data/featured_data.parquet")
    
    if cache_file.exists() and not force_refresh:
        logger.info("Loading cached featured data")
        featured_data = pd.read_parquet(cache_file)
        logger.info(f"Loaded cached data: {len(featured_data)} observations, {featured_data['ticker'].nunique()} tickers")
    else:
        logger.info("Processing raw data and engineering features")
        
        # Load and process raw data
        raw_data = load_and_process_data(config_path, force_refresh)
        
        # Engineer features
        engine = FeatureEngine(config_path)
        featured_data = engine.engineer_features(raw_data)
        
        # Cache the result
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        featured_data.to_parquet(cache_file)
        logger.info(f"Featured data cached to {cache_file}")
    
    # Get feature columns
    feature_cols = [col for col in featured_data.columns if col.endswith('_zscore')]
    logger.info(f"Dataset summary:")
    logger.info(f"  Total observations: {len(featured_data):,}")
    logger.info(f"  Unique tickers: {featured_data['ticker'].nunique()}")
    logger.info(f"  Date range: {featured_data['Date'].min().date()} to {featured_data['Date'].max().date()}")
    logger.info(f"  Features engineered: {len(feature_cols)}")
    
    return featured_data


def train_models(featured_data: pd.DataFrame, config_path: str, 
                skip_training: bool = False, force_refresh: bool = False) -> ModelEnsemble:
    """Train or load model ensemble."""
    logger.info("=" * 60)
    logger.info("STEP 2: MODEL TRAINING")
    logger.info("=" * 60)
    
    model_dir = Path("models")
    ensemble_info_file = model_dir / "ensemble_info.pkl"
    
    # Check if trained models exist
    if ensemble_info_file.exists() and not force_refresh and not skip_training:
        logger.info("Loading existing model ensemble")
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        ensemble = ModelEnsemble(config)
        ensemble.load(str(model_dir))
        logger.info(f"Loaded ensemble with weights: {ensemble.weights}")
        
    elif skip_training:
        logger.warning("Skip training requested but no existing models found")
        raise FileNotFoundError("No existing models found. Please train models first.")
        
    else:
        logger.info("Training new model ensemble")
        
        # Prepare training data
        engine = FeatureEngine(config_path)
        
        # Use data up to 2020 for training, rest for out-of-sample testing
        train_end = '2020-01-01'
        X, y = engine.get_feature_matrix(
            featured_data, 
            use_zscore=True,
            start_date='2005-01-01',
            end_date=train_end
        )
        
        if len(X) == 0:
            raise ValueError("No training data available")
        
        logger.info(f"Training data: {X.shape[0]} observations, {X.shape[1]} features")
        
        # Load configuration and train ensemble
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        ensemble = ModelEnsemble(config)
        blend_method = config['portfolio'].get('signal_blend', 'equal')
        ensemble.fit(X, y, blend_method=blend_method)
        
        # Save trained models
        model_dir.mkdir(parents=True, exist_ok=True)
        ensemble.save(str(model_dir))
        
        logger.info(f"Model training completed with weights: {ensemble.weights}")
    
    return ensemble


def run_backtest_analysis(featured_data: pd.DataFrame, 
                         ensemble: ModelEnsemble,
                         config_path: str,
                         benchmark_ticker: str = "SPY") -> dict:
    """Run backtesting and performance analysis."""
    logger.info("=" * 60)
    logger.info("STEP 3: BACKTESTING AND ANALYSIS")
    logger.info("=" * 60)
    
    # Initialize portfolio optimizer
    portfolio_optimizer = PortfolioOptimizer(config_path)
    
    # Run complete backtest
    results = run_full_backtest(
        featured_data, 
        ensemble, 
        portfolio_optimizer, 
        config_path, 
        benchmark_ticker
    )
    
    # Extract key metrics
    if 'error' not in results['backtest_results']:
        metrics = results['backtest_results']['metrics']
        
        logger.info("BACKTEST RESULTS SUMMARY:")
        logger.info("-" * 30)
        logger.info(f"Total Return: {metrics['total_return']:.1%}")
        logger.info(f"Annualized Return: {metrics['annualized_return']:.1%}")
        logger.info(f"Annual Volatility: {metrics['annual_volatility']:.1%}")
        logger.info(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        logger.info(f"Maximum Drawdown: {metrics['max_drawdown']:.1%}")
        logger.info(f"Hit Rate: {metrics['hit_rate']:.1%}")
        logger.info(f"Total Periods: {metrics['total_periods']}")
        
        # Benchmark comparison if available
        if 'information_ratio' in metrics:
            logger.info(f"Information Ratio: {metrics['information_ratio']:.2f}")
            logger.info(f"Tracking Error: {metrics['tracking_error']:.1%}")
    
    return results


def save_final_results(results: dict, config_path: str):
    """Save final results and weights."""
    logger.info("=" * 60)
    logger.info("STEP 4: SAVING RESULTS")
    logger.info("=" * 60)
    
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save latest portfolio weights if available
    if 'backtest_results' in results and 'positions_history' in results['backtest_results']:
        positions_history = results['backtest_results']['positions_history']
        
        if positions_history:
            latest_positions = positions_history[-1]['positions']
            latest_date = positions_history[-1]['date']
            
            # Convert to DataFrame
            weights_df = pd.DataFrame({
                'ticker': latest_positions.index,
                'position': latest_positions.values,
                'date': latest_date
            })
            
            weights_file = results_dir / "weights_latest.csv"
            weights_df.to_csv(weights_file, index=False)
            logger.info(f"Latest portfolio weights saved to {weights_file}")
    
    # Save performance summary
    if 'backtest_results' in results and 'metrics' in results['backtest_results']:
        metrics = results['backtest_results']['metrics']
        
        summary_file = results_dir / "performance_summary.yaml"
        with open(summary_file, 'w') as f:
            yaml.dump(metrics, f, default_flow_style=False)
        
        logger.info(f"Performance summary saved to {summary_file}")
    
    logger.info("Results saved successfully")


def print_final_summary(results: dict):
    """Print final summary of the pipeline execution."""
    logger.info("=" * 60)
    logger.info("PIPELINE EXECUTION COMPLETE")
    logger.info("=" * 60)
    
    logger.info("Generated outputs:")
    logger.info("📊 Reports:")
    logger.info("  - reports/tearsheet.html (QuantStats tearsheet)")
    logger.info("  - reports/performance_summary.txt")
    
    logger.info("📈 Plots:")
    logger.info("  - plots/equity_curve.png")
    logger.info("  - plots/drawdown_chart.png") 
    logger.info("  - plots/rolling_metrics.png")
    logger.info("  - plots/monthly_heatmap.png")
    
    logger.info("📋 Results:")
    logger.info("  - results/weights_latest.csv")
    logger.info("  - results/performance_summary.yaml")
    
    if 'backtest_results' in results and 'metrics' in results['backtest_results']:
        metrics = results['backtest_results']['metrics']
        logger.info("")
        logger.info("🎯 KEY PERFORMANCE METRICS:")
        logger.info(f"   Total Return: {metrics.get('total_return', 0):.1%}")
        logger.info(f"   Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
        logger.info(f"   Max Drawdown: {metrics.get('max_drawdown', 0):.1%}")
    
    logger.info("")
    logger.info("✅ Multi-Factor Equity Alpha Engine pipeline completed successfully!")


def main():
    """Main pipeline execution function."""
    args = parse_arguments()
    
    # Print welcome message
    logger.info("🚀 MULTI-FACTOR EQUITY ALPHA ENGINE")
    logger.info("=" * 60)
    logger.info(f"Universe: {args.universe}")
    logger.info(f"Backtest Period: {args.start} to {args.end}")
    logger.info(f"Initial Capital: ${args.capital:,.0f}")
    logger.info(f"Benchmark: {args.benchmark}")
    
    try:
        # Update configuration
        config = update_config(args, args.config)
        
        # Step 1: Load/prepare data
        featured_data = load_or_prepare_data(args.config, args.force_refresh)
        
        # Step 2: Train models
        ensemble = train_models(featured_data, args.config, args.skip_training, args.force_refresh)
        
        # Step 3: Run backtest
        results = run_backtest_analysis(featured_data, ensemble, args.config, args.benchmark)
        
        # Step 4: Save results
        save_final_results(results, args.config)
        
        # Print final summary
        print_final_summary(results)
        
        # Update build log
        with open("project_build_log.txt", "a") as f:
            f.write(f"\n✅ Pipeline executed successfully at {datetime.now()}")
            f.write(f"\n   Universe: {args.universe}, Period: {args.start} to {args.end}")
            if 'backtest_results' in results and 'metrics' in results['backtest_results']:
                metrics = results['backtest_results']['metrics']
                f.write(f"\n   Performance: {metrics.get('annualized_return', 0):.1%} return, {metrics.get('sharpe_ratio', 0):.2f} Sharpe")
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        
        # Update build log with error
        with open("project_build_log.txt", "a") as f:
            f.write(f"\n❌ Pipeline failed at {datetime.now()}: {str(e)}")
        
        raise


if __name__ == "__main__":
    main()
