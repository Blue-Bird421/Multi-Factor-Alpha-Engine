"""
Machine learning models module for the Alpha Engine.

Implements Ridge Regression, XGBoost, and Neural Network models
for predicting stock returns.
"""

import pandas as pd
import numpy as np
import yaml
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
import xgboost as xgb
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import joblib
from pathlib import Path
import logging
from typing import Dict, List, Optional, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)


class BaseModel:
    """Base class for all models."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize base model.
        
        Args:
            config: Model configuration dictionary
        """
        self.config = config
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'BaseModel':
        """
        Fit the model.
        
        Args:
            X: Features DataFrame
            y: Target Series
            
        Returns:
            Self for method chaining
        """
        raise NotImplementedError("Subclasses must implement fit method")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Features DataFrame
            
        Returns:
            Predictions array
        """
        raise NotImplementedError("Subclasses must implement predict method")
    
    def save(self, filepath: str) -> None:
        """Save model to disk."""
        raise NotImplementedError("Subclasses must implement save method")
    
    def load(self, filepath: str) -> None:
        """Load model from disk."""
        raise NotImplementedError("Subclasses must implement load method")


class RidgeModel(BaseModel):
    """Ridge Regression model for return prediction."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Ridge model.
        
        Args:
            config: Model configuration dictionary
        """
        super().__init__(config)
        model_config = config.get('ridge', {})
        
        self.model = Ridge(
            alpha=model_config.get('alpha', 1.0),
            fit_intercept=model_config.get('fit_intercept', True),
            random_state=42
        )
        
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'RidgeModel':
        """
        Fit Ridge regression model.
        
        Args:
            X: Features DataFrame
            y: Target Series
            
        Returns:
            Self for method chaining
        """
        logger.info("Fitting Ridge regression model")
        
        # Handle missing values
        X_clean = X.fillna(X.median())
        y_clean = y.fillna(y.median())
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_clean)
        
        # Fit model
        self.model.fit(X_scaled, y_clean)
        self.is_fitted = True
        
        # Calculate feature importance (absolute coefficients)
        self.feature_importance_ = pd.Series(
            np.abs(self.model.coef_),
            index=X.columns,
            name='importance'
        ).sort_values(ascending=False)
        
        logger.info(f"Ridge model fitted with {len(X.columns)} features")
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions with Ridge model.
        
        Args:
            X: Features DataFrame
            
        Returns:
            Predictions array
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        X_clean = X.fillna(X.median())
        X_scaled = self.scaler.transform(X_clean)
        return self.model.predict(X_scaled)
    
    def get_feature_importance(self) -> pd.Series:
        """Get feature importance scores."""
        if not hasattr(self, 'feature_importance_'):
            raise ValueError("Model must be fitted to get feature importance")
        return self.feature_importance_
    
    def save(self, filepath: str) -> None:
        """Save Ridge model to disk."""
        save_dict = {
            'model': self.model,
            'scaler': self.scaler,
            'config': self.config,
            'is_fitted': self.is_fitted
        }
        if hasattr(self, 'feature_importance_'):
            save_dict['feature_importance_'] = self.feature_importance_
        
        joblib.dump(save_dict, filepath)
        logger.info(f"Ridge model saved to {filepath}")
    
    def load(self, filepath: str) -> None:
        """Load Ridge model from disk."""
        save_dict = joblib.load(filepath)
        self.model = save_dict['model']
        self.scaler = save_dict['scaler']
        self.config = save_dict['config']
        self.is_fitted = save_dict['is_fitted']
        
        if 'feature_importance_' in save_dict:
            self.feature_importance_ = save_dict['feature_importance_']
        
        logger.info(f"Ridge model loaded from {filepath}")


class XGBoostModel(BaseModel):
    """XGBoost model for return prediction."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize XGBoost model.
        
        Args:
            config: Model configuration dictionary
        """
        super().__init__(config)
        model_config = config.get('xgboost', {})
        
        self.model_params = {
            'objective': 'reg:squarederror',
            'n_estimators': model_config.get('n_estimators', 100),
            'max_depth': model_config.get('max_depth', 6),
            'learning_rate': model_config.get('learning_rate', 0.1),
            'random_state': model_config.get('random_state', 42),
            'n_jobs': -1
        }
        
        self.model = xgb.XGBRegressor(**self.model_params)
        
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'XGBoostModel':
        """
        Fit XGBoost model.
        
        Args:
            X: Features DataFrame
            y: Target Series
            
        Returns:
            Self for method chaining
        """
        logger.info("Fitting XGBoost model")
        
        # Handle missing values
        X_clean = X.fillna(X.median())
        y_clean = y.fillna(y.median())
        
        # XGBoost can handle raw features, but we'll still scale for consistency
        X_scaled = self.scaler.fit_transform(X_clean)
        
        # Fit model
        self.model.fit(X_scaled, y_clean)
        self.is_fitted = True
        
        # Get feature importance
        self.feature_importance_ = pd.Series(
            self.model.feature_importances_,
            index=X.columns,
            name='importance'
        ).sort_values(ascending=False)
        
        logger.info(f"XGBoost model fitted with {len(X.columns)} features")
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions with XGBoost model.
        
        Args:
            X: Features DataFrame
            
        Returns:
            Predictions array
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        X_clean = X.fillna(X.median())
        X_scaled = self.scaler.transform(X_clean)
        return self.model.predict(X_scaled)
    
    def get_feature_importance(self) -> pd.Series:
        """Get feature importance scores."""
        if not hasattr(self, 'feature_importance_'):
            raise ValueError("Model must be fitted to get feature importance")
        return self.feature_importance_
    
    def save(self, filepath: str) -> None:
        """Save XGBoost model to disk."""
        save_dict = {
            'model': self.model,
            'scaler': self.scaler,
            'config': self.config,
            'is_fitted': self.is_fitted
        }
        if hasattr(self, 'feature_importance_'):
            save_dict['feature_importance_'] = self.feature_importance_
        
        joblib.dump(save_dict, filepath)
        logger.info(f"XGBoost model saved to {filepath}")
    
    def load(self, filepath: str) -> None:
        """Load XGBoost model from disk."""
        save_dict = joblib.load(filepath)
        self.model = save_dict['model']
        self.scaler = save_dict['scaler']
        self.config = save_dict['config']
        self.is_fitted = save_dict['is_fitted']
        
        if 'feature_importance_' in save_dict:
            self.feature_importance_ = save_dict['feature_importance_']
        
        logger.info(f"XGBoost model loaded from {filepath}")


class NeuralNetworkModel(BaseModel):
    """Feed-forward Neural Network model for return prediction."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Neural Network model.
        
        Args:
            config: Model configuration dictionary
        """
        super().__init__(config)
        self.model_config = config.get('neural_network', {})
        
        self.hidden_layers = self.model_config.get('hidden_layers', [64, 32])
        self.dropout_rate = self.model_config.get('dropout_rate', 0.2)
        self.learning_rate = self.model_config.get('learning_rate', 0.001)
        self.epochs = self.model_config.get('epochs', 100)
        self.batch_size = self.model_config.get('batch_size', 256)
        
    def _build_model(self, input_dim: int) -> keras.Model:
        """
        Build neural network architecture.
        
        Args:
            input_dim: Number of input features
            
        Returns:
            Compiled Keras model
        """
        model = keras.Sequential()
        
        # Input layer
        model.add(layers.Dense(self.hidden_layers[0], 
                              activation='relu', 
                              input_dim=input_dim))
        model.add(layers.Dropout(self.dropout_rate))
        
        # Hidden layers
        for units in self.hidden_layers[1:]:
            model.add(layers.Dense(units, activation='relu'))
            model.add(layers.Dropout(self.dropout_rate))
        
        # Output layer
        model.add(layers.Dense(1, activation='linear'))
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'NeuralNetworkModel':
        """
        Fit Neural Network model.
        
        Args:
            X: Features DataFrame
            y: Target Series
            
        Returns:
            Self for method chaining
        """
        logger.info("Fitting Neural Network model")
        
        # Handle missing values
        X_clean = X.fillna(X.median())
        y_clean = y.fillna(y.median())
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_clean)
        
        # Build model
        self.model = self._build_model(X_scaled.shape[1])
        
        # Split data for validation
        val_split = 0.2
        split_idx = int(len(X_scaled) * (1 - val_split))
        
        X_train, X_val = X_scaled[:split_idx], X_scaled[split_idx:]
        y_train, y_val = y_clean.iloc[:split_idx], y_clean.iloc[split_idx:]
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6
            )
        ]
        
        # Train model
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=callbacks,
            verbose=0
        )
        
        self.is_fitted = True
        self.training_history = history
        
        # Calculate feature importance using permutation importance
        self._calculate_feature_importance(X_clean, y_clean)
        
        logger.info(f"Neural Network model fitted with {len(X.columns)} features")
        return self
    
    def _calculate_feature_importance(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Calculate feature importance using permutation importance."""
        X_scaled = self.scaler.transform(X.fillna(X.median()))
        baseline_score = self.model.evaluate(X_scaled, y, verbose=0)[0]  # MSE
        
        importances = []
        for i in range(X_scaled.shape[1]):
            # Permute feature i
            X_permuted = X_scaled.copy()
            X_permuted[:, i] = np.random.permutation(X_permuted[:, i])
            
            # Calculate score with permuted feature
            permuted_score = self.model.evaluate(X_permuted, y, verbose=0)[0]
            
            # Importance is increase in error
            importance = permuted_score - baseline_score
            importances.append(importance)
        
        self.feature_importance_ = pd.Series(
            importances,
            index=X.columns,
            name='importance'
        ).sort_values(ascending=False)
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions with Neural Network model.
        
        Args:
            X: Features DataFrame
            
        Returns:
            Predictions array
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        X_clean = X.fillna(X.median())
        X_scaled = self.scaler.transform(X_clean)
        predictions = self.model.predict(X_scaled, verbose=0)
        return predictions.flatten()
    
    def get_feature_importance(self) -> pd.Series:
        """Get feature importance scores."""
        if not hasattr(self, 'feature_importance_'):
            raise ValueError("Model must be fitted to get feature importance")
        return self.feature_importance_
    
    def save(self, filepath: str) -> None:
        """Save Neural Network model to disk."""
        model_path = filepath.replace('.pkl', '_model.h5')
        scaler_path = filepath.replace('.pkl', '_scaler.pkl')
        config_path = filepath.replace('.pkl', '_config.pkl')
        
        # Save Keras model
        self.model.save(model_path)
        
        # Save scaler and config
        joblib.dump(self.scaler, scaler_path)
        
        save_dict = {
            'config': self.config,
            'model_config': self.model_config,
            'is_fitted': self.is_fitted
        }
        if hasattr(self, 'feature_importance_'):
            save_dict['feature_importance_'] = self.feature_importance_
        
        joblib.dump(save_dict, config_path)
        logger.info(f"Neural Network model saved to {filepath}")
    
    def load(self, filepath: str) -> None:
        """Load Neural Network model from disk."""
        model_path = filepath.replace('.pkl', '_model.h5')
        scaler_path = filepath.replace('.pkl', '_scaler.pkl')
        config_path = filepath.replace('.pkl', '_config.pkl')
        
        # Load Keras model
        self.model = keras.models.load_model(model_path)
        
        # Load scaler and config
        self.scaler = joblib.load(scaler_path)
        save_dict = joblib.load(config_path)
        
        self.config = save_dict['config']
        self.model_config = save_dict['model_config']
        self.is_fitted = save_dict['is_fitted']
        
        if 'feature_importance_' in save_dict:
            self.feature_importance_ = save_dict['feature_importance_']
        
        logger.info(f"Neural Network model loaded from {filepath}")


class ModelEnsemble:
    """Ensemble of multiple models for improved predictions."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize model ensemble.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.models = {}
        self.weights = None
        self.is_fitted = False
        
        # Initialize models
        self.models['ridge'] = RidgeModel(config)
        self.models['xgboost'] = XGBoostModel(config)
        self.models['neural_network'] = NeuralNetworkModel(config)
        
    def fit(self, X: pd.DataFrame, y: pd.Series, 
            blend_method: str = 'equal') -> 'ModelEnsemble':
        """
        Fit all models in the ensemble.
        
        Args:
            X: Features DataFrame
            y: Target Series
            blend_method: Method for blending predictions ('equal' or 'ir_weighted')
            
        Returns:
            Self for method chaining
        """
        logger.info("Fitting model ensemble")
        
        self.model_scores = {}
        
        # Fit each model
        for name, model in self.models.items():
            logger.info(f"Fitting {name} model")
            try:
                model.fit(X, y)
                
                # Calculate out-of-sample score using time series split
                tscv = TimeSeriesSplit(n_splits=3)
                scores = []
                
                for train_idx, val_idx in tscv.split(X):
                    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                    
                    # Create temporary model for validation
                    temp_model = type(model)(self.config)
                    temp_model.fit(X_train, y_train)
                    predictions = temp_model.predict(X_val)
                    
                    # Calculate information ratio (mean return / std of returns)
                    mean_pred = np.mean(predictions)
                    std_pred = np.std(predictions) if np.std(predictions) > 0 else 1e-6
                    ir = mean_pred / std_pred if std_pred > 0 else 0
                    scores.append(abs(ir))  # Use absolute IR for weighting
                
                self.model_scores[name] = np.mean(scores)
                logger.info(f"{name} average IR: {self.model_scores[name]:.4f}")
                
            except Exception as e:
                logger.error(f"Failed to fit {name} model: {e}")
                self.model_scores[name] = 0
        
        # Calculate ensemble weights
        if blend_method == 'equal':
            self.weights = {name: 1/len(self.models) for name in self.models}
        elif blend_method == 'ir_weighted':
            total_score = sum(self.model_scores.values())
            if total_score > 0:
                self.weights = {name: score/total_score for name, score in self.model_scores.items()}
            else:
                self.weights = {name: 1/len(self.models) for name in self.models}
        
        self.is_fitted = True
        logger.info(f"Ensemble weights: {self.weights}")
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make ensemble predictions.
        
        Args:
            X: Features DataFrame
            
        Returns:
            Ensemble predictions array
        """
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before making predictions")
        
        predictions = {}
        for name, model in self.models.items():
            if model.is_fitted:
                try:
                    predictions[name] = model.predict(X)
                except Exception as e:
                    logger.warning(f"Failed to get predictions from {name}: {e}")
                    predictions[name] = np.zeros(len(X))
        
        # Weighted average of predictions
        ensemble_pred = np.zeros(len(X))
        for name, pred in predictions.items():
            ensemble_pred += self.weights[name] * pred
        
        return ensemble_pred
    
    def get_model(self, name: str) -> BaseModel:
        """Get individual model from ensemble."""
        return self.models.get(name)
    
    def save(self, model_dir: str) -> None:
        """Save all models in ensemble."""
        model_path = Path(model_dir)
        model_path.mkdir(parents=True, exist_ok=True)
        
        for name, model in self.models.items():
            if model.is_fitted:
                filepath = model_path / f"{name}_model.pkl"
                model.save(str(filepath))
        
        # Save ensemble metadata
        ensemble_info = {
            'weights': self.weights,
            'model_scores': self.model_scores,
            'is_fitted': self.is_fitted
        }
        joblib.dump(ensemble_info, model_path / "ensemble_info.pkl")
        
        logger.info(f"Ensemble saved to {model_dir}")
    
    def load(self, model_dir: str) -> None:
        """Load all models in ensemble."""
        model_path = Path(model_dir)
        
        for name, model in self.models.items():
            filepath = model_path / f"{name}_model.pkl"
            if filepath.exists():
                model.load(str(filepath))
        
        # Load ensemble metadata
        ensemble_info_path = model_path / "ensemble_info.pkl"
        if ensemble_info_path.exists():
            ensemble_info = joblib.load(ensemble_info_path)
            self.weights = ensemble_info['weights']
            self.model_scores = ensemble_info['model_scores']
            self.is_fitted = ensemble_info['is_fitted']
        
        logger.info(f"Ensemble loaded from {model_dir}")


def train_model(model_type: str, X: pd.DataFrame, y: pd.Series, 
                config_path: str = "config.yaml") -> BaseModel:
    """
    Convenience function to train a single model.
    
    Args:
        model_type: Type of model ('ridge', 'xgboost', 'neural_network')
        X: Features DataFrame
        y: Target Series
        config_path: Path to configuration file
        
    Returns:
        Trained model
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    if model_type == 'ridge':
        model = RidgeModel(config)
    elif model_type == 'xgboost':
        model = XGBoostModel(config)
    elif model_type == 'neural_network':
        model = NeuralNetworkModel(config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model.fit(X, y)


if __name__ == "__main__":
    # Example usage
    from .data import load_and_process_data
    from .features import FeatureEngine
    
    # Load and prepare data
    data = load_and_process_data()
    engine = FeatureEngine()
    featured_data = engine.engineer_features(data)
    X, y = engine.get_feature_matrix(featured_data, start_date="2010-01-01", end_date="2020-01-01")
    
    # Train ensemble
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    ensemble = ModelEnsemble(config)
    ensemble.fit(X, y)
    
    print("Ensemble training complete")
    print(f"Model weights: {ensemble.weights}")
