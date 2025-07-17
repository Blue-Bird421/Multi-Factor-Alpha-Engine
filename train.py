#!/usr/bin/env python3
"""
Training script for individual models in the Alpha Engine.

Usage:
    python train.py --model ridge
    python train.py --model xgboost  
    python train.py --model neural_network
"""

import argparse
import yaml
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from alpha_engine.data import load_and_process_data
from alpha_engine.features import FeatureEngine
from alpha_engine.models import RidgeModel, XGBoostModel, NeuralNetworkModel, ModelEnsemble

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train individual models for the Alpha Engine')
    
    parser.add_argument('--model', type=str, required=True,
                       choices=['ridge', 'xgboost', 'neural_network', 'ensemble'],
                       help='Model type to train')
    
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file')
    
    parser.add_argument('--data-refresh', action='store_true',
                       help='Force refresh of data from source')
    
    parser.add_argument('--train-start', type=str, default='2005-01-01',
                       help='Training start date (YYYY-MM-DD)')
    
    parser.add_argument('--train-end', type=str, default='2020-01-01',
                       help='Training end date (YYYY-MM-DD)')
    
    parser.add_argument('--save-model', action='store_true', default=True,
                       help='Save trained model to disk')
    
    return parser.parse_args()


def load_training_data(config_path: str, force_refresh: bool = False) -> pd.DataFrame:
    """Load and process training data."""
    logger.info("Loading and processing training data")
    
    # Load raw data
    data = load_and_process_data(config_path, force_refresh)
    
    # Engineer features
    engine = FeatureEngine(config_path)
    featured_data = engine.engineer_features(data)
    
    logger.info(f"Training data loaded: {len(featured_data)} observations, {featured_data['ticker'].nunique()} tickers")
    
    return featured_data


def prepare_training_features(featured_data: pd.DataFrame, 
                            start_date: str, 
                            end_date: str) -> tuple:
    """Prepare feature matrix and target variable for training."""
    logger.info(f"Preparing training features from {start_date} to {end_date}")
    
    engine = FeatureEngine()
    X, y = engine.get_feature_matrix(
        featured_data, 
        use_zscore=True,
        start_date=start_date,
        end_date=end_date
    )
    
    logger.info(f"Training set: {X.shape[0]} observations, {X.shape[1]} features")
    logger.info(f"Target statistics: mean={y.mean():.4f}, std={y.std():.4f}")
    
    return X, y


def train_single_model(model_type: str, X: pd.DataFrame, y: pd.Series, config_path: str):
    """Train a single model."""
    logger.info(f"Training {model_type} model")
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize model
    if model_type == 'ridge':
        model = RidgeModel(config)
    elif model_type == 'xgboost':
        model = XGBoostModel(config)
    elif model_type == 'neural_network':
        model = NeuralNetworkModel(config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Train model
    model.fit(X, y)
    
    # Print feature importance
    if hasattr(model, 'get_feature_importance'):
        importance = model.get_feature_importance().head(10)
        logger.info(f"Top 10 features for {model_type}:")
        for feature, imp in importance.items():
            logger.info(f"  {feature}: {imp:.4f}")
    
    return model


def train_ensemble_model(X: pd.DataFrame, y: pd.Series, config_path: str):
    """Train ensemble of all models."""
    logger.info("Training ensemble model")
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize ensemble
    ensemble = ModelEnsemble(config)
    
    # Train ensemble
    blend_method = config['portfolio'].get('signal_blend', 'equal')
    ensemble.fit(X, y, blend_method=blend_method)
    
    logger.info(f"Ensemble training complete with weights: {ensemble.weights}")
    
    return ensemble


def save_model(model, model_type: str, output_dir: str = "models"):
    """Save trained model to disk."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if model_type == 'ensemble':
        model.save(str(output_path))
        logger.info(f"Ensemble saved to {output_path}")
    else:
        model_file = output_path / f"{model_type}_model.pkl"
        model.save(str(model_file))
        logger.info(f"{model_type} model saved to {model_file}")


def evaluate_model(model, X: pd.DataFrame, y: pd.Series, model_type: str):
    """Evaluate model performance on training data."""
    logger.info(f"Evaluating {model_type} model")
    
    # Make predictions
    predictions = model.predict(X)
    
    # Calculate metrics
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    
    mse = mean_squared_error(y, predictions)
    mae = mean_absolute_error(y, predictions)
    correlation = np.corrcoef(y, predictions)[0, 1]
    
    # Information Coefficient (IC)
    ic = correlation
    
    # Hit rate (percentage of correct directional predictions)
    y_direction = np.sign(y)
    pred_direction = np.sign(predictions)
    hit_rate = (y_direction == pred_direction).mean()
    
    logger.info(f"{model_type} Performance Metrics:")
    logger.info(f"  MSE: {mse:.6f}")
    logger.info(f"  MAE: {mae:.6f}")
    logger.info(f"  Correlation (IC): {ic:.4f}")
    logger.info(f"  Hit Rate: {hit_rate:.1%}")
    
    return {
        'mse': mse,
        'mae': mae,
        'ic': ic,
        'hit_rate': hit_rate
    }


def main():
    """Main training function."""
    args = parse_arguments()
    
    logger.info(f"Starting training for {args.model} model")
    logger.info(f"Configuration: {args.config}")
    logger.info(f"Training period: {args.train_start} to {args.train_end}")
    
    try:
        # Load training data
        featured_data = load_training_data(args.config, args.data_refresh)
        
        # Prepare features
        X, y = prepare_training_features(featured_data, args.train_start, args.train_end)
        
        if len(X) == 0:
            logger.error("No training data available")
            return
        
        # Train model
        if args.model == 'ensemble':
            model = train_ensemble_model(X, y, args.config)
        else:
            model = train_single_model(args.model, X, y, args.config)
        
        # Evaluate model
        metrics = evaluate_model(model, X, y, args.model)
        
        # Save model
        if args.save_model:
            save_model(model, args.model)
        
        logger.info(f"Training completed successfully for {args.model}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()
