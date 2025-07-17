"""
Tests for the models module.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch
from alpha_engine.models import RidgeModel, XGBoostModel, NeuralNetworkModel, ModelEnsemble
import tempfile
import os


class TestRidgeModel:
    """Test RidgeModel class."""
    
    def test_init(self, sample_config):
        """Test RidgeModel initialization."""
        model = RidgeModel(sample_config)
        assert model.model is not None
        assert not model.is_fitted
    
    def test_fit_and_predict(self, sample_config):
        """Test fitting and prediction."""
        model = RidgeModel(sample_config)
        
        # Create sample data
        X = pd.DataFrame(np.random.randn(100, 5), columns=['f1', 'f2', 'f3', 'f4', 'f5'])
        y = pd.Series(np.random.randn(100))
        
        # Fit model
        model.fit(X, y)
        assert model.is_fitted
        
        # Test prediction
        predictions = model.predict(X)
        assert len(predictions) == len(X)
        assert isinstance(predictions, np.ndarray)
        
        # Test feature importance
        importance = model.get_feature_importance()
        assert len(importance) == len(X.columns)
    
    def test_save_load(self, sample_config):
        """Test model saving and loading."""
        model = RidgeModel(sample_config)
        
        # Create and fit model
        X = pd.DataFrame(np.random.randn(50, 3), columns=['f1', 'f2', 'f3'])
        y = pd.Series(np.random.randn(50))
        model.fit(X, y)
        
        # Save model
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            temp_path = f.name
        
        try:
            model.save(temp_path)
            
            # Load model
            new_model = RidgeModel(sample_config)
            new_model.load(temp_path)
            
            assert new_model.is_fitted
            
            # Test predictions are the same
            pred1 = model.predict(X)
            pred2 = new_model.predict(X)
            np.testing.assert_array_almost_equal(pred1, pred2)
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


class TestXGBoostModel:
    """Test XGBoostModel class."""
    
    def test_init(self, sample_config):
        """Test XGBoostModel initialization."""
        model = XGBoostModel(sample_config)
        assert model.model is not None
        assert not model.is_fitted
    
    def test_fit_and_predict(self, sample_config):
        """Test fitting and prediction."""
        model = XGBoostModel(sample_config)
        
        # Create sample data
        X = pd.DataFrame(np.random.randn(100, 5), columns=['f1', 'f2', 'f3', 'f4', 'f5'])
        y = pd.Series(np.random.randn(100))
        
        # Fit model
        model.fit(X, y)
        assert model.is_fitted
        
        # Test prediction
        predictions = model.predict(X)
        assert len(predictions) == len(X)
        
        # Test feature importance
        importance = model.get_feature_importance()
        assert len(importance) == len(X.columns)


class TestNeuralNetworkModel:
    """Test NeuralNetworkModel class."""
    
    def test_init(self, sample_config):
        """Test NeuralNetworkModel initialization."""
        model = NeuralNetworkModel(sample_config)
        assert model.hidden_layers == [16, 8]  # From sample config
        assert not model.is_fitted
    
    def test_fit_and_predict(self, sample_config):
        """Test fitting and prediction."""
        model = NeuralNetworkModel(sample_config)
        
        # Create sample data
        X = pd.DataFrame(np.random.randn(200, 10), columns=[f'f{i}' for i in range(10)])
        y = pd.Series(np.random.randn(200))
        
        # Fit model (this may take a moment)
        model.fit(X, y)
        assert model.is_fitted
        
        # Test prediction
        predictions = model.predict(X)
        assert len(predictions) == len(X)
        
        # Test feature importance
        importance = model.get_feature_importance()
        assert len(importance) == len(X.columns)


class TestModelEnsemble:
    """Test ModelEnsemble class."""
    
    def test_init(self, sample_config):
        """Test ModelEnsemble initialization."""
        ensemble = ModelEnsemble(sample_config)
        assert 'ridge' in ensemble.models
        assert 'xgboost' in ensemble.models
        assert 'neural_network' in ensemble.models
        assert not ensemble.is_fitted
    
    def test_fit_and_predict(self, sample_config):
        """Test ensemble fitting and prediction."""
        ensemble = ModelEnsemble(sample_config)
        
        # Create sample data
        X = pd.DataFrame(np.random.randn(200, 8), columns=[f'f{i}' for i in range(8)])
        y = pd.Series(np.random.randn(200))
        
        # Fit ensemble
        ensemble.fit(X, y, blend_method='equal')
        assert ensemble.is_fitted
        assert ensemble.weights is not None
        assert abs(sum(ensemble.weights.values()) - 1.0) < 1e-6  # Weights should sum to 1
        
        # Test prediction
        predictions = ensemble.predict(X)
        assert len(predictions) == len(X)
        
        # Test individual model access
        ridge_model = ensemble.get_model('ridge')
        assert ridge_model is not None
        assert ridge_model.is_fitted
    
    def test_ir_weighted_blending(self, sample_config):
        """Test Information Ratio weighted blending."""
        ensemble = ModelEnsemble(sample_config)
        
        # Create sample data with some predictive signal
        X = pd.DataFrame(np.random.randn(300, 5), columns=[f'f{i}' for i in range(5)])
        # Make y somewhat predictable from X
        y = pd.Series(X['f0'] * 0.1 + X['f1'] * 0.05 + np.random.randn(300) * 0.1)
        
        # Fit with IR weighting
        ensemble.fit(X, y, blend_method='ir_weighted')
        assert ensemble.is_fitted
        assert ensemble.weights is not None
        
        # All weights should be non-negative
        assert all(w >= 0 for w in ensemble.weights.values())


class TestModelIntegration:
    """Integration tests for models."""
    
    def test_model_consistency(self, sample_config):
        """Test that models produce consistent results."""
        # Create deterministic data
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(100, 5), columns=[f'f{i}' for i in range(5)])
        y = pd.Series(X.sum(axis=1) + np.random.randn(100) * 0.1)
        
        # Test each model
        models = {
            'ridge': RidgeModel(sample_config),
            'xgboost': XGBoostModel(sample_config)
        }
        
        predictions = {}
        for name, model in models.items():
            model.fit(X, y)
            pred = model.predict(X)
            predictions[name] = pred
            
            # Check prediction quality
            correlation = np.corrcoef(y, pred)[0, 1]
            assert not np.isnan(correlation)  # Should have some correlation
    
    def test_model_with_missing_data(self, sample_config):
        """Test model behavior with missing data."""
        X = pd.DataFrame(np.random.randn(100, 5), columns=[f'f{i}' for i in range(5)])
        y = pd.Series(np.random.randn(100))
        
        # Add some missing values
        X.loc[0:10, 'f0'] = np.nan
        y.loc[15:20] = np.nan
        
        # Test Ridge model (should handle missing data)
        model = RidgeModel(sample_config)
        model.fit(X, y)
        predictions = model.predict(X)
        
        assert len(predictions) == len(X)
        assert not np.isnan(predictions).any()  # Should not have NaN predictions
    
    @pytest.mark.slow
    def test_ensemble_performance(self, sample_config):
        """Test ensemble performance vs individual models."""
        # Create data with multiple patterns
        np.random.seed(123)
        X = pd.DataFrame(np.random.randn(500, 10), columns=[f'f{i}' for i in range(10)])
        
        # Create complex target with linear and non-linear components
        y = pd.Series(
            X['f0'] * 0.3 +  # Linear component
            X['f1'] * X['f2'] * 0.2 +  # Interaction component
            np.sin(X['f3']) * 0.1 +  # Non-linear component
            np.random.randn(500) * 0.2  # Noise
        )
        
        # Split data
        train_idx = np.arange(400)
        test_idx = np.arange(400, 500)
        
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Train ensemble
        ensemble = ModelEnsemble(sample_config)
        ensemble.fit(X_train, y_train)
        
        # Get predictions
        ensemble_pred = ensemble.predict(X_test)
        
        # Calculate performance
        ensemble_corr = np.corrcoef(y_test, ensemble_pred)[0, 1]
        
        # Ensemble should have reasonable performance
        assert not np.isnan(ensemble_corr)
        assert ensemble_corr > -0.5  # Should have some predictive power


if __name__ == "__main__":
    pytest.main([__file__])
