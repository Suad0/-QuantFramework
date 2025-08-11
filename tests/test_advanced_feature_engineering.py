"""
Tests for advanced feature engineering capabilities
"""
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings

from src.application.services.feature_engine import (
    FeatureEngine, 
    RollingWindowProcessor, 
    DataImputer, 
    LookAheadBiasDetector,
    FeatureSelector,
    FeatureValidationReport
)
from src.domain.exceptions import ValidationError


class TestRollingWindowProcessor:
    """Test rolling window operations"""
    
    def setup_method(self):
        self.processor = RollingWindowProcessor()
        
        # Create sample data
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        np.random.seed(42)
        self.sample_data = pd.DataFrame({
            'close': 100 + np.cumsum(np.random.randn(100) * 0.5),
            'volume': np.random.randint(1000, 10000, 100),
            'high': 100 + np.cumsum(np.random.randn(100) * 0.5) + np.random.rand(100) * 2,
            'low': 100 + np.cumsum(np.random.randn(100) * 0.5) - np.random.rand(100) * 2
        }, index=dates)
    
    def test_basic_rolling_operations(self):
        """Test basic rolling window operations"""
        config = {
            'windows': [10, 20],
            'operations': ['mean', 'std'],
            'columns': ['close']
        }
        
        result = self.processor.apply_rolling_operations(self.sample_data, config)
        
        # Check that new columns were created
        expected_columns = [
            'close_rolling_10_mean', 'close_rolling_10_std',
            'close_rolling_20_mean', 'close_rolling_20_std'
        ]
        
        for col in expected_columns:
            assert col in result.columns
        
        # Check that rolling mean is calculated correctly
        expected_mean_10 = self.sample_data['close'].rolling(10).mean()
        pd.testing.assert_series_equal(
            result['close_rolling_10_mean'], 
            expected_mean_10, 
            check_names=False
        )
    
    def test_advanced_rolling_operations(self):
        """Test advanced rolling operations like RSI and momentum"""
        config = {
            'windows': [14],
            'operations': ['rsi', 'momentum', 'volatility'],
            'columns': ['close']
        }
        
        result = self.processor.apply_rolling_operations(self.sample_data, config)
        
        # Check that advanced indicators were calculated
        assert 'close_rolling_14_rsi' in result.columns
        assert 'close_rolling_14_momentum' in result.columns
        assert 'close_rolling_14_volatility' in result.columns
        
        # RSI should be between 0 and 100
        rsi_values = result['close_rolling_14_rsi'].dropna()
        assert all(0 <= val <= 100 for val in rsi_values)
    
    def test_quantile_operations(self):
        """Test quantile rolling operations"""
        config = {
            'windows': [20],
            'operations': ['quantile'],
            'columns': ['close'],
            'quantiles': [0.25, 0.75]
        }
        
        result = self.processor.apply_rolling_operations(self.sample_data, config)
        
        assert 'close_rolling_20_quantile_25' in result.columns
        assert 'close_rolling_20_quantile_75' in result.columns
    
    def test_insufficient_data_warning(self):
        """Test warning when data length is less than window size"""
        small_data = self.sample_data.head(5)
        config = {
            'windows': [10],
            'operations': ['mean'],
            'columns': ['close']
        }
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = self.processor.apply_rolling_operations(small_data, config)
            assert len(w) > 0
            assert "Data length" in str(w[0].message)


class TestDataImputer:
    """Test data imputation strategies"""
    
    def setup_method(self):
        self.imputer = DataImputer()
        
        # Create sample data with missing values
        dates = pd.date_range('2020-01-01', periods=50, freq='D')
        np.random.seed(42)
        data = pd.DataFrame({
            'price': 100 + np.cumsum(np.random.randn(50) * 0.5),
            'volume': np.random.randint(1000, 10000, 50),
            'returns': np.random.randn(50) * 0.02
        }, index=dates)
        
        # Introduce missing values
        data.loc[data.index[5:8], 'price'] = np.nan
        data.loc[data.index[15:17], 'volume'] = np.nan
        data.loc[data.index[25], 'returns'] = np.nan
        
        self.sample_data_with_missing = data
    
    def test_forward_fill_imputation(self):
        """Test forward fill imputation"""
        result = self.imputer.impute_missing_data(
            self.sample_data_with_missing, 
            'forward_fill'
        )
        
        # Check that missing values were filled
        assert result.isnull().sum().sum() < self.sample_data_with_missing.isnull().sum().sum()
        
        # Check that forward fill was applied correctly
        assert result.loc[result.index[6], 'price'] == result.loc[result.index[4], 'price']
    
    def test_linear_interpolation(self):
        """Test linear interpolation"""
        result = self.imputer.impute_missing_data(
            self.sample_data_with_missing, 
            'linear_interpolation'
        )
        
        # Check that missing values were filled
        assert result.isnull().sum().sum() < self.sample_data_with_missing.isnull().sum().sum()
    
    def test_mean_imputation(self):
        """Test mean imputation"""
        result = self.imputer.impute_missing_data(
            self.sample_data_with_missing, 
            'mean_imputation'
        )
        
        # Check that missing values were filled with mean
        original_mean = self.sample_data_with_missing['price'].mean()
        filled_values = result.loc[self.sample_data_with_missing['price'].isna(), 'price']
        assert all(abs(val - original_mean) < 1e-10 for val in filled_values)
    
    def test_knn_imputation(self):
        """Test KNN imputation"""
        result = self.imputer.impute_missing_data(
            self.sample_data_with_missing, 
            'knn_imputation',
            n_neighbors=3
        )
        
        # Check that missing values were filled
        assert result.isnull().sum().sum() == 0
    
    def test_seasonal_imputation(self):
        """Test seasonal imputation"""
        # Create data with seasonal pattern
        dates = pd.date_range('2020-01-01', periods=500, freq='D')
        seasonal_data = pd.DataFrame({
            'value': 100 + 10 * np.sin(2 * np.pi * np.arange(500) / 365) + np.random.randn(500)
        }, index=dates)
        
        # Introduce missing values
        seasonal_data.loc[seasonal_data.index[100:105], 'value'] = np.nan
        
        result = self.imputer.impute_missing_data(
            seasonal_data, 
            'seasonal_imputation',
            period=365
        )
        
        # Check that missing values were filled
        assert result.isnull().sum().sum() == 0
    
    def test_carry_forward_with_decay(self):
        """Test carry forward with decay imputation"""
        result = self.imputer.impute_missing_data(
            self.sample_data_with_missing, 
            'carry_forward_with_decay',
            decay_rate=0.9
        )
        
        # Check that missing values were filled
        assert result.isnull().sum().sum() < self.sample_data_with_missing.isnull().sum().sum()


class TestLookAheadBiasDetector:
    """Test look-ahead bias detection"""
    
    def setup_method(self):
        self.detector = LookAheadBiasDetector()
        
        # Create sample data
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        np.random.seed(42)
        self.original_data = pd.DataFrame({
            'close': 100 + np.cumsum(np.random.randn(100) * 0.5),
            'volume': np.random.randint(1000, 10000, 100)
        }, index=dates)
    
    def test_detect_suspicious_stability(self):
        """Test detection of suspiciously stable features"""
        # Create features with suspicious stability (potential future data usage)
        features = self.original_data.copy()
        features['suspicious_feature'] = 50.0  # Constant value
        
        report = self.detector.detect_bias(self.original_data, features)
        
        # Should detect the suspicious stability
        assert len(report.bias_details) > 0 or len(report.warnings) > 0
    
    def test_detect_perfect_correlation(self):
        """Test detection of perfect correlations"""
        features = self.original_data.copy()
        # Create a feature that's perfectly correlated (shifted future data)
        features['future_close'] = self.original_data['close'].shift(-1)
        
        report = self.detector.detect_bias(self.original_data, features)
        
        # Should detect high correlation
        assert report.look_ahead_bias_detected or len(report.warnings) > 0
    
    def test_prevent_bias_configuration(self):
        """Test bias prevention in configuration"""
        config = {
            'rolling_operations': [
                {'window': 10, 'operation': 'mean', 'center': True}
            ]
        }
        
        safe_config = self.detector.prevent_bias(self.original_data, config)
        
        # Should modify configuration to prevent bias
        assert 'temporal_constraints' in safe_config
        assert safe_config['temporal_constraints']['max_future_periods'] == 0


class TestFeatureSelector:
    """Test feature selection and dimensionality reduction"""
    
    def setup_method(self):
        self.selector = FeatureSelector()
        
        # Create sample features and target
        np.random.seed(42)
        n_samples = 100
        n_features = 20
        
        # Create features with varying importance
        X = np.random.randn(n_samples, n_features)
        # Make some features more predictive
        true_coefficients = np.zeros(n_features)
        true_coefficients[:5] = [2, 1.5, 1, 0.5, 0.3]
        
        y = X @ true_coefficients + np.random.randn(n_samples) * 0.1
        
        self.features = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(n_features)])
        self.target = pd.Series(y, name='target')
    
    def test_univariate_f_selection(self):
        """Test univariate F-test feature selection"""
        selected_features, feature_names = self.selector.select_features(
            self.features, 
            self.target, 
            method='univariate_f',
            n_features=10
        )
        
        assert len(feature_names) == 10
        assert selected_features.shape[1] == 10
        
        # Most important features should be selected
        assert 'feature_0' in feature_names  # Highest coefficient
    
    def test_mutual_info_selection(self):
        """Test mutual information feature selection"""
        selected_features, feature_names = self.selector.select_features(
            self.features, 
            self.target, 
            method='mutual_info',
            n_features=8
        )
        
        assert len(feature_names) == 8
        assert selected_features.shape[1] == 8
    
    def test_rfe_selection(self):
        """Test recursive feature elimination"""
        selected_features, feature_names = self.selector.select_features(
            self.features, 
            self.target, 
            method='rfe',
            n_features=5
        )
        
        assert len(feature_names) == 5
        assert selected_features.shape[1] == 5
    
    def test_variance_threshold_selection(self):
        """Test variance threshold selection"""
        # Add a constant feature
        features_with_constant = self.features.copy()
        features_with_constant['constant_feature'] = 1.0
        
        selected_features, feature_names = self.selector.select_features(
            features_with_constant, 
            self.target, 
            method='variance_threshold',
            n_features=15,
            threshold=0.01
        )
        
        # Constant feature should be removed
        assert 'constant_feature' not in feature_names
    
    def test_correlation_filter(self):
        """Test correlation filter"""
        # Add highly correlated features
        features_with_corr = self.features.copy()
        features_with_corr['corr_feature'] = self.features['feature_0'] + np.random.randn(len(self.features)) * 0.01
        
        selected_features, feature_names = self.selector.select_features(
            features_with_corr, 
            self.target, 
            method='correlation_filter',
            n_features=15,
            threshold=0.95
        )
        
        # One of the highly correlated features should be removed
        assert not ('feature_0' in feature_names and 'corr_feature' in feature_names)
    
    def test_pca_reduction(self):
        """Test PCA dimensionality reduction"""
        reduced_features, pca_model = self.selector.reduce_dimensions(
            self.features, 
            method='pca',
            n_components=5
        )
        
        assert reduced_features.shape[1] == 5
        assert reduced_features.columns.tolist() == ['PC1', 'PC2', 'PC3', 'PC4', 'PC5']
        
        # Check that PCA model was fitted
        assert hasattr(pca_model, 'explained_variance_ratio_')
    
    def test_ica_reduction(self):
        """Test ICA dimensionality reduction"""
        reduced_features, ica_model = self.selector.reduce_dimensions(
            self.features, 
            method='ica',
            n_components=5
        )
        
        assert reduced_features.shape[1] == 5
        assert reduced_features.columns.tolist() == ['IC1', 'IC2', 'IC3', 'IC4', 'IC5']


class TestFeatureEngine:
    """Test the main FeatureEngine class with advanced capabilities"""
    
    def setup_method(self):
        self.engine = FeatureEngine()
        
        # Create sample data
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        np.random.seed(42)
        self.sample_data = pd.DataFrame({
            'close': 100 + np.cumsum(np.random.randn(100) * 0.5),
            'high': 100 + np.cumsum(np.random.randn(100) * 0.5) + np.random.rand(100) * 2,
            'low': 100 + np.cumsum(np.random.randn(100) * 0.5) - np.random.rand(100) * 2,
            'volume': np.random.randint(1000, 10000, 100)
        }, index=dates)
        
        # Introduce some missing values
        self.sample_data.loc[self.sample_data.index[10:13], 'close'] = np.nan
    
    def test_comprehensive_feature_engineering(self):
        """Test comprehensive feature engineering pipeline"""
        config = {
            'imputation': {
                'enabled': True,
                'strategy': 'linear_interpolation'
            },
            'rolling_operations': {
                'enabled': True,
                'windows': [10, 20],
                'operations': ['mean', 'std', 'rsi'],
                'columns': ['close']
            },
            'indicators': {
                'ADX': {'period': 14}
            },
            'bias_prevention': {
                'enabled': True
            }
        }
        
        result = self.engine.compute_features(self.sample_data, config)
        
        # Check that imputation was applied
        assert result['close'].isnull().sum() == 0
        
        # Check that rolling operations were applied
        assert 'close_rolling_10_mean' in result.columns
        assert 'close_rolling_20_std' in result.columns
        assert 'close_rolling_10_rsi' in result.columns
    
    def test_feature_selection_integration(self):
        """Test feature selection integration"""
        # First create some features
        config = {
            'rolling_operations': {
                'enabled': True,
                'windows': [5, 10, 20],
                'operations': ['mean', 'std'],
                'columns': ['close']
            }
        }
        
        features = self.engine.compute_features(self.sample_data, config)
        
        # Add a target column (future returns)
        features['target'] = features['close'].pct_change(fill_method=None).shift(-1)
        
        # Now apply feature selection
        config_with_selection = config.copy()
        config_with_selection['feature_selection'] = {
            'enabled': True,
            'target_column': 'target',
            'method': 'univariate_f',
            'n_features': 3
        }
        
        result = self.engine.compute_features(features.drop('target', axis=1), config_with_selection)
        
        # Should have selected features plus any original columns
        assert len(result.columns) <= 10  # Should be reduced
    
    def test_dimensionality_reduction_integration(self):
        """Test dimensionality reduction integration"""
        config = {
            'rolling_operations': {
                'enabled': True,
                'windows': [5, 10, 20],
                'operations': ['mean', 'std', 'min', 'max'],
                'columns': ['close', 'volume']
            },
            'dimensionality_reduction': {
                'enabled': True,
                'method': 'pca',
                'n_components': 5
            }
        }
        
        result = self.engine.compute_features(self.sample_data, config)
        
        # Should have PCA components
        pca_columns = [col for col in result.columns if col.startswith('PC')]
        assert len(pca_columns) == 5
    
    def test_feature_validation_with_bias_detection(self):
        """Test feature validation including bias detection"""
        features = self.sample_data.copy()
        
        # Add a suspicious feature (future data)
        features['future_close'] = features['close'].shift(-1)
        
        report = self.engine.validate_features(features, self.sample_data)
        
        # Should detect potential issues
        assert isinstance(report, FeatureValidationReport)
        assert 'total_features' in report.statistics
        assert 'memory_usage_mb' in report.statistics
    
    def test_feature_importance_calculation(self):
        """Test feature importance calculation"""
        # Create features
        config = {
            'rolling_operations': {
                'enabled': True,
                'windows': [10, 20],
                'operations': ['mean', 'std'],
                'columns': ['close']
            }
        }
        
        features = self.engine.compute_features(self.sample_data, config)
        
        # Create target
        target = features['close'].pct_change(fill_method=None).shift(-1).dropna()
        
        # Calculate importance
        importance = self.engine.get_feature_importance(
            features.drop('close', axis=1), 
            target, 
            method='random_forest'
        )
        
        assert isinstance(importance, pd.Series)
        assert len(importance) > 0
        assert all(importance >= 0)  # Importance scores should be non-negative
    
    def test_feature_summary_creation(self):
        """Test comprehensive feature summary"""
        features = self.engine.compute_features(self.sample_data, {
            'rolling_operations': {
                'enabled': True,
                'windows': [10],
                'operations': ['mean'],
                'columns': ['close']
            }
        })
        
        summary = self.engine.create_feature_summary(features)
        
        assert 'total_features' in summary
        assert 'numeric_features' in summary
        assert 'missing_data_percentage' in summary
        assert 'feature_types' in summary
        
        # Check feature-level details
        for col in features.columns:
            assert col in summary['feature_types']
            assert 'dtype' in summary['feature_types'][col]
            assert 'missing_count' in summary['feature_types'][col]


if __name__ == '__main__':
    pytest.main([__file__])