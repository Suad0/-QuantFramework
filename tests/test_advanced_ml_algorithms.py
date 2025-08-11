"""
Tests for advanced ML algorithms implementation.

This module tests the enhanced LSTM with attention mechanisms,
XGBoost, Random Forest, and SVM implementations with hyperparameter
optimization and regularization features.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
import warnings

from src.infrastructure.ml.advanced_algorithms import (
    EnhancedXGBoostRegressor,
    EnhancedRandomForestRegressor,
    EnhancedSVMRegressor
)
from src.infrastructure.ml.lstm_wrapper import LSTMWrapper
from src.infrastructure.ml.hyperparameter_optimization import (
    HyperparameterOptimizer,
    GridSearchOptimizer,
    RandomSearchOptimizer,
    RegularizationManager
)
from src.infrastructure.ml.ml_framework import MLFramework


class TestEnhancedXGBoostRegressor:
    """Test enhanced XGBoost regressor."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample financial data."""
        np.random.seed(42)
        n_samples = 100
        n_features = 5
        
        X = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        y = pd.Series(
            np.random.randn(n_samples) * 0.1 + X.sum(axis=1) * 0.1,
            name='target'
        )
        
        return X, y
    
    def test_basic_functionality(self, sample_data):
        """Test basic XGBoost functionality."""
        X, y = sample_data
        
        model = EnhancedXGBoostRegressor(
            n_estimators=10,
            max_depth=3,
            random_state=42
        )
        
        # Test fitting
        model.fit(X, y)
        assert model.is_fitted_
        assert model.feature_importances_ is not None
        
        # Test prediction
        predictions = model.predict(X)
        assert len(predictions) == len(X)
        assert isinstance(predictions, np.ndarray)
    
    def test_hyperparameter_tuning(self, sample_data):
        """Test hyperparameter tuning functionality."""
        X, y = sample_data
        
        model = EnhancedXGBoostRegressor(
            auto_tune=True,
            tune_method='grid',
            cv_folds=3,
            random_state=42
        )
        
        # Mock XGBoost to avoid dependency issues in tests
        with patch('src.infrastructure.ml.advanced_algorithms.xgb') as mock_xgb:
            mock_model = Mock()
            mock_model.fit.return_value = mock_model
            mock_model.predict.return_value = np.random.randn(len(X))
            mock_model.feature_importances_ = np.random.rand(len(X.columns))
            
            mock_xgb.XGBRegressor.return_value = mock_model
            
            model.fit(X, y)
            
            assert model.best_params_ is not None
            assert model.is_fitted_
    
    def test_uncertainty_prediction(self, sample_data):
        """Test uncertainty prediction."""
        X, y = sample_data
        
        model = EnhancedXGBoostRegressor(random_state=42)
        
        with patch('src.infrastructure.ml.advanced_algorithms.xgb') as mock_xgb:
            mock_model = Mock()
            mock_model.fit.return_value = mock_model
            mock_model.predict.return_value = np.random.randn(len(X))
            mock_model.feature_importances_ = np.random.rand(len(X.columns))
            
            mock_xgb.XGBRegressor.return_value = mock_model
            
            model.fit(X, y)
            predictions, uncertainties = model.predict_with_uncertainty(X)
            
            assert len(predictions) == len(X)
            assert len(uncertainties) == len(X)
            assert all(u >= 0 for u in uncertainties)


class TestEnhancedRandomForestRegressor:
    """Test enhanced Random Forest regressor."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample financial data."""
        np.random.seed(42)
        n_samples = 100
        n_features = 5
        
        X = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        y = pd.Series(
            np.random.randn(n_samples) * 0.1 + X.sum(axis=1) * 0.1,
            name='target'
        )
        
        return X, y
    
    def test_basic_functionality(self, sample_data):
        """Test basic Random Forest functionality."""
        X, y = sample_data
        
        model = EnhancedRandomForestRegressor(
            n_estimators=10,
            max_depth=3,
            random_state=42
        )
        
        # Test fitting
        model.fit(X, y)
        assert model.is_fitted_
        assert model.feature_importances_ is not None
        
        # Test prediction
        predictions = model.predict(X)
        assert len(predictions) == len(X)
        assert isinstance(predictions, np.ndarray)
    
    def test_feature_selection(self, sample_data):
        """Test feature selection functionality."""
        X, y = sample_data
        
        model = EnhancedRandomForestRegressor(
            n_estimators=10,
            feature_selection=True,
            feature_selection_threshold=0.1,
            random_state=42
        )
        
        model.fit(X, y)
        
        # Should have selected some features
        if model.selected_features_ is not None:
            assert len(model.selected_features_) <= len(X.columns)
    
    def test_uncertainty_prediction(self, sample_data):
        """Test uncertainty prediction using tree variance."""
        X, y = sample_data
        
        model = EnhancedRandomForestRegressor(
            n_estimators=10,
            random_state=42
        )
        
        model.fit(X, y)
        predictions, uncertainties = model.predict_with_uncertainty(X)
        
        assert len(predictions) == len(X)
        assert len(uncertainties) == len(X)
        assert all(u >= 0 for u in uncertainties)


class TestEnhancedSVMRegressor:
    """Test enhanced SVM regressor."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample financial data."""
        np.random.seed(42)
        n_samples = 50  # Smaller dataset for SVM
        n_features = 3
        
        X = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        y = pd.Series(
            np.random.randn(n_samples) * 0.1 + X.sum(axis=1) * 0.1,
            name='target'
        )
        
        return X, y
    
    def test_basic_functionality(self, sample_data):
        """Test basic SVM functionality."""
        X, y = sample_data
        
        model = EnhancedSVMRegressor(
            C=1.0,
            kernel='rbf',
            random_state=42
        )
        
        # Test fitting
        model.fit(X, y)
        assert model.is_fitted_
        
        # Test prediction
        predictions = model.predict(X)
        assert len(predictions) == len(X)
        assert isinstance(predictions, np.ndarray)
    
    def test_feature_scaling(self, sample_data):
        """Test feature scaling functionality."""
        X, y = sample_data
        
        model = EnhancedSVMRegressor(
            scale_features=True,
            random_state=42
        )
        
        model.fit(X, y)
        
        # Should have created a scaler
        assert model.scaler_ is not None
        
        # Test prediction with scaling
        predictions = model.predict(X)
        assert len(predictions) == len(X)


class TestLSTMWithAttention:
    """Test LSTM with attention mechanisms."""
    
    @pytest.fixture
    def sample_time_series_data(self):
        """Create sample time series data."""
        np.random.seed(42)
        n_samples = 200
        n_features = 3
        sequence_length = 60
        
        # Create time series data
        data = []
        for i in range(n_samples):
            sample = np.random.randn(n_features) + np.sin(i * 0.1)
            data.append(sample)
        
        X = pd.DataFrame(data, columns=[f'feature_{i}' for i in range(n_features)])
        y = pd.Series(np.random.randn(n_samples) * 0.1 + X.sum(axis=1) * 0.1)
        
        return X, y
    
    def test_attention_lstm_creation(self, sample_time_series_data):
        """Test LSTM with attention creation."""
        X, y = sample_time_series_data
        
        model = LSTMWrapper(
            sequence_length=30,
            lstm_units=32,
            use_attention=True,
            epochs=1,  # Quick test
            random_state=42
        )
        
        # Test parameter access
        params = model.get_params()
        assert 'use_attention' in params
        assert 'attention_heads' in params
        assert 'use_multihead_attention' in params
    
    def test_multihead_attention_lstm(self, sample_time_series_data):
        """Test LSTM with multi-head attention."""
        X, y = sample_time_series_data
        
        model = LSTMWrapper(
            sequence_length=30,
            lstm_units=32,
            use_multihead_attention=True,
            attention_heads=4,
            epochs=1,  # Quick test
            random_state=42
        )
        
        params = model.get_params()
        assert params['use_multihead_attention'] is True
        assert params['attention_heads'] == 4
    
    def test_regularization_parameters(self, sample_time_series_data):
        """Test regularization parameters."""
        X, y = sample_time_series_data
        
        model = LSTMWrapper(
            sequence_length=30,
            lstm_units=32,
            l1_reg=0.01,
            l2_reg=0.01,
            use_layer_norm=True,
            use_residual=True,
            epochs=1,  # Quick test
            random_state=42
        )
        
        params = model.get_params()
        assert params['l1_reg'] == 0.01
        assert params['l2_reg'] == 0.01
        assert params['use_layer_norm'] is True
        assert params['use_residual'] is True


class TestHyperparameterOptimization:
    """Test hyperparameter optimization framework."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data."""
        np.random.seed(42)
        n_samples = 50
        n_features = 3
        
        X = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        y = pd.Series(np.random.randn(n_samples))
        
        return X, y
    
    def test_grid_search_optimizer(self, sample_data):
        """Test grid search optimizer."""
        X, y = sample_data
        
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(random_state=42)
        
        param_space = {
            'n_estimators': [10, 20],
            'max_depth': [3, 5]
        }
        
        optimizer = GridSearchOptimizer(cv_folds=3)
        best_params, best_score = optimizer.optimize(model, param_space, X, y)
        
        assert best_params is not None
        assert 'n_estimators' in best_params
        assert 'max_depth' in best_params
        assert isinstance(best_score, float)
    
    def test_random_search_optimizer(self, sample_data):
        """Test random search optimizer."""
        X, y = sample_data
        
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(random_state=42)
        
        param_space = {
            'n_estimators': [10, 20, 30],
            'max_depth': [3, 5, 7]
        }
        
        optimizer = RandomSearchOptimizer(n_iter=5, cv_folds=3, random_state=42)
        best_params, best_score = optimizer.optimize(model, param_space, X, y)
        
        assert best_params is not None
        assert isinstance(best_score, float)
    
    def test_hyperparameter_optimizer_main_class(self, sample_data):
        """Test main hyperparameter optimizer class."""
        X, y = sample_data
        
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(random_state=42)
        
        param_space = {
            'n_estimators': [10, 20],
            'max_depth': [3, 5]
        }
        
        optimizer = HyperparameterOptimizer()
        best_params, best_score, optimization_info = optimizer.optimize(
            model, param_space, X, y, method='grid', cv_folds=3
        )
        
        assert best_params is not None
        assert isinstance(best_score, float)
        assert 'method' in optimization_info
        assert 'optimization_time' in optimization_info
    
    def test_default_param_spaces(self):
        """Test default parameter spaces."""
        optimizer = HyperparameterOptimizer()
        
        # Test XGBoost parameter space
        xgb_params = optimizer.get_default_param_space('xgboost')
        assert 'n_estimators' in xgb_params
        assert 'max_depth' in xgb_params
        assert 'learning_rate' in xgb_params
        
        # Test Random Forest parameter space
        rf_params = optimizer.get_default_param_space('random_forest')
        assert 'n_estimators' in rf_params
        assert 'max_depth' in rf_params
        
        # Test SVM parameter space
        svm_params = optimizer.get_default_param_space('svm')
        assert 'C' in svm_params
        assert 'kernel' in svm_params


class TestMLFrameworkIntegration:
    """Test ML framework integration with enhanced algorithms."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data."""
        np.random.seed(42)
        n_samples = 100
        n_features = 5
        
        X = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        y = pd.Series(np.random.randn(n_samples))
        
        return X, y
    
    def test_enhanced_algorithms_in_framework(self, sample_data):
        """Test that enhanced algorithms are available in framework."""
        X, y = sample_data
        
        framework = MLFramework()
        
        # Test enhanced algorithms are registered
        assert 'enhanced_xgboost' in framework.supported_algorithms
        assert 'enhanced_random_forest' in framework.supported_algorithms
        assert 'enhanced_svm' in framework.supported_algorithms
        assert 'lstm_attention' in framework.supported_algorithms
    
    def test_hyperparameter_optimization_in_framework(self, sample_data):
        """Test hyperparameter optimization through framework."""
        X, y = sample_data
        
        framework = MLFramework()
        
        config = {
            'algorithm': 'random_forest',
            'hyperparameter_optimization': {
                'enabled': True,
                'method': 'grid',
                'param_space': {
                    'n_estimators': [10, 20],
                    'max_depth': [3, 5]
                },
                'optimizer_kwargs': {
                    'cv_folds': 3
                }
            },
            'register_model': False
        }
        
        model = framework.train_model(X, y, config)
        
        assert model is not None
        assert hasattr(model, 'predict')
    
    def test_regularization_in_framework(self, sample_data):
        """Test regularization through framework."""
        X, y = sample_data
        
        framework = MLFramework()
        
        config = {
            'algorithm': 'random_forest',
            'regularization': {
                'techniques': ['early_stopping'],
                'kwargs': {}
            },
            'register_model': False
        }
        
        model = framework.train_model(X, y, config)
        
        assert model is not None
        assert hasattr(model, 'predict')


class TestRegularizationManager:
    """Test regularization manager."""
    
    def test_regularization_manager_creation(self):
        """Test regularization manager creation."""
        manager = RegularizationManager()
        
        assert manager is not None
        assert hasattr(manager, 'apply_regularization')
        assert hasattr(manager, 'techniques')
    
    def test_apply_regularization(self):
        """Test applying regularization techniques."""
        from sklearn.ensemble import RandomForestRegressor
        
        manager = RegularizationManager()
        model = RandomForestRegressor()
        
        # Test applying regularization (should not fail)
        regularized_model = manager.apply_regularization(
            model, ['early_stopping']
        )
        
        assert regularized_model is not None


if __name__ == '__main__':
    pytest.main([__file__])