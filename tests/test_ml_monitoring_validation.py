"""
Tests for ML model monitoring and validation components.

This module tests the uncertainty quantification, financial metrics,
drift detection, and A/B testing components.
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta
import warnings

from src.infrastructure.ml.uncertainty_quantification import (
    UncertaintyQuantificationManager,
    BootstrapUncertaintyQuantifier,
    EnsembleUncertaintyQuantifier
)
from src.infrastructure.ml.financial_metrics import FinancialMetricsCalculator
from src.infrastructure.ml.drift_detection import (
    ModelDriftMonitor,
    StatisticalDriftDetector,
    PerformanceDriftDetector,
    DriftType
)
from src.infrastructure.ml.ab_testing import ABTester, StatisticalTest
from src.domain.exceptions import ValidationError


class TestUncertaintyQuantification:
    """Test uncertainty quantification functionality."""
    
    def setup_method(self):
        """Set up test data."""
        np.random.seed(42)
        
        # Create synthetic financial data
        n_samples = 200
        n_features = 5
        
        self.X = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        
        # Create target with some noise
        self.y = pd.Series(
            self.X.sum(axis=1) + np.random.randn(n_samples) * 0.1,
            name='target'
        )
        
        # Train a simple model
        self.model = RandomForestRegressor(n_estimators=10, random_state=42)
        self.model.fit(self.X, self.y)
        
        self.uq_manager = UncertaintyQuantificationManager()
    
    def test_bootstrap_uncertainty_quantification(self):
        """Test bootstrap uncertainty quantification."""
        result = self.uq_manager.quantify_uncertainty(
            self.model, self.X, self.y, method='bootstrap', n_bootstrap=50
        )
        
        assert 'predictions' in result
        assert 'uncertainty' in result
        assert 'confidence_lower' in result
        assert 'confidence_upper' in result
        assert len(result['predictions']) == len(self.X)
        assert len(result['uncertainty']) == len(self.X)
        assert result['confidence_level'] == 0.95
        assert result['n_bootstrap'] <= 50  # Some might fail
    
    def test_ensemble_uncertainty_quantification(self):
        """Test ensemble-based uncertainty quantification."""
        result = self.uq_manager.quantify_uncertainty(
            self.model, self.X, method='ensemble'
        )
        
        assert 'predictions' in result
        assert 'uncertainty' in result
        assert 'confidence_lower' in result
        assert 'confidence_upper' in result
        assert len(result['predictions']) == len(self.X)
        assert result['n_estimators'] == 10  # RandomForest with 10 estimators
    
    def test_prediction_intervals(self):
        """Test prediction interval calculation."""
        result = self.uq_manager.quantify_uncertainty(
            self.model, self.X, self.y, method='bootstrap', n_bootstrap=20
        )
        
        intervals = self.uq_manager.get_prediction_intervals(
            result, confidence_levels=[0.68, 0.95]
        )
        
        assert 0.68 in intervals
        assert 0.95 in intervals
        
        # Check that 95% interval is wider than 68%
        width_68 = intervals[0.68][1] - intervals[0.68][0]
        width_95 = intervals[0.95][1] - intervals[0.95][0]
        assert np.all(width_95 >= width_68)
    
    def test_invalid_method(self):
        """Test error handling for invalid method."""
        with pytest.raises(ValidationError):
            self.uq_manager.quantify_uncertainty(
                self.model, self.X, self.y, method='invalid_method'
            )


class TestFinancialMetrics:
    """Test financial metrics calculation."""
    
    def setup_method(self):
        """Set up test data."""
        np.random.seed(42)
        
        # Create synthetic return data
        n_samples = 252  # One year of daily data
        
        # True returns (with some trend and volatility)
        self.y_true = pd.Series(
            np.random.randn(n_samples) * 0.02 + 0.0005,  # Daily returns
            name='returns'
        )
        
        # Predicted returns (with some correlation to true)
        noise = np.random.randn(n_samples) * 0.01
        self.y_pred = pd.Series(
            self.y_true * 0.7 + noise,  # 70% correlation with noise
            name='predicted_returns'
        )
        
        # Price series
        self.prices = pd.Series(
            100 * (1 + self.y_true).cumprod(),
            name='prices'
        )
        
        self.metrics_calc = FinancialMetricsCalculator()
    
    def test_directional_accuracy(self):
        """Test directional accuracy calculation."""
        result = self.metrics_calc.calculate_directional_accuracy(
            self.y_true, self.y_pred, method='returns'
        )
        
        assert 'directional_accuracy' in result
        assert 'hit_rate' in result
        assert 'up_accuracy' in result
        assert 'down_accuracy' in result
        assert 0 <= result['directional_accuracy'] <= 1
        assert result['total_predictions'] > 0
        assert result['method'] == 'returns'
    
    def test_profit_based_metrics(self):
        """Test profit-based metrics calculation."""
        result = self.metrics_calc.calculate_profit_based_metrics(
            self.y_true, self.y_pred, self.prices
        )
        
        assert 'total_return' in result
        assert 'annualized_return' in result
        assert 'volatility' in result
        assert 'sharpe_ratio' in result
        assert 'max_drawdown' in result
        assert 'win_rate' in result
        assert 'profit_factor' in result
        
        # Check that volatility is positive
        assert result['volatility'] >= 0
        
        # Check that max drawdown is negative or zero
        assert result['max_drawdown'] <= 0
    
    def test_risk_adjusted_metrics(self):
        """Test risk-adjusted metrics calculation."""
        # Create benchmark returns
        benchmark_returns = pd.Series(
            np.random.randn(len(self.y_true)) * 0.015 + 0.0003
        )
        
        result = self.metrics_calc.calculate_risk_adjusted_metrics(
            self.y_true, self.y_pred, benchmark_returns
        )
        
        assert 'sortino_ratio' in result
        assert 'information_ratio' in result
        assert 'tracking_error' in result
        assert 'var_95' in result
        assert 'var_99' in result
        assert 'cvar_95' in result
        assert 'cvar_99' in result
        assert 'skewness' in result
        assert 'kurtosis' in result
        
        # VaR should be negative (loss)
        assert result['var_95'] <= 0
        assert result['var_99'] <= 0
        
        # CVaR should be more negative than VaR
        assert result['cvar_95'] <= result['var_95']
        assert result['cvar_99'] <= result['var_99']
    
    def test_comprehensive_metrics(self):
        """Test comprehensive metrics calculation."""
        result = self.metrics_calc.calculate_comprehensive_metrics(
            self.y_true, self.y_pred, self.prices
        )
        
        assert 'directional' in result
        assert 'profit' in result
        assert 'risk_adjusted' in result
        assert 'prediction_quality' in result
        assert 'timestamp' in result
        assert 'n_observations' in result
        
        assert result['n_observations'] == len(self.y_true)


class TestDriftDetection:
    """Test drift detection functionality."""
    
    def setup_method(self):
        """Set up test data."""
        np.random.seed(42)
        
        # Create baseline data
        n_baseline = 200
        n_features = 3
        
        self.baseline_features = pd.DataFrame(
            np.random.randn(n_baseline, n_features),
            columns=['feature_1', 'feature_2', 'feature_3']
        )
        
        self.baseline_targets = pd.Series(
            self.baseline_features.sum(axis=1) + np.random.randn(n_baseline) * 0.1,
            name='target'
        )
        
        # Train model on baseline data
        self.model = LinearRegression()
        self.model.fit(self.baseline_features, self.baseline_targets)
        self.baseline_predictions = pd.Series(
            self.model.predict(self.baseline_features)
        )
        
        # Calculate baseline performance
        self.baseline_performance = {
            'mse': ((self.baseline_targets - self.baseline_predictions) ** 2).mean(),
            'mae': abs(self.baseline_targets - self.baseline_predictions).mean(),
            'correlation': self.baseline_targets.corr(self.baseline_predictions)
        }
        
        self.drift_monitor = ModelDriftMonitor()
        self.drift_monitor.set_baseline(
            self.baseline_features,
            self.baseline_targets,
            self.baseline_predictions,
            self.baseline_performance
        )
    
    def test_statistical_drift_detection(self):
        """Test statistical drift detection."""
        detector = StatisticalDriftDetector()
        
        # Create drifted data (shifted distribution)
        drifted_data = pd.Series(np.random.randn(100) + 2.0)  # Shifted by 2
        baseline_data = pd.Series(np.random.randn(100))
        
        # Test Kolmogorov-Smirnov
        ks_result = detector.kolmogorov_smirnov_test(baseline_data, drifted_data)
        assert 'drift_detected' in ks_result
        assert 'p_value' in ks_result
        assert 'statistic' in ks_result
        assert ks_result['drift_detected'] == True  # Should detect drift
        
        # Test Mann-Whitney
        mw_result = detector.mann_whitney_test(baseline_data, drifted_data)
        assert 'drift_detected' in mw_result
        assert mw_result['drift_detected'] == True  # Should detect drift
    
    def test_performance_drift_detection(self):
        """Test performance drift detection."""
        detector = PerformanceDriftDetector()
        
        # Create degraded performance data
        degraded_targets = self.baseline_targets + np.random.randn(len(self.baseline_targets)) * 0.5
        degraded_predictions = self.baseline_predictions + np.random.randn(len(self.baseline_predictions)) * 0.3
        
        result = detector.detect_performance_drift(
            degraded_targets, degraded_predictions, self.baseline_performance
        )
        
        assert 'drift_detected' in result
        assert 'metrics' in result
        assert 'current_performance' in result
        assert 'baseline_performance' in result
    
    def test_comprehensive_drift_monitoring(self):
        """Test comprehensive drift monitoring."""
        # Create current data with some drift
        current_features = self.baseline_features.copy()
        current_features['feature_1'] += 1.0  # Add drift to feature 1
        
        current_targets = pd.Series(
            current_features.sum(axis=1) + np.random.randn(len(current_features)) * 0.2
        )
        
        current_predictions = pd.Series(
            self.model.predict(current_features)
        )
        
        result = self.drift_monitor.monitor_drift(
            current_features, current_targets, current_predictions
        )
        
        assert 'timestamp' in result
        assert 'data_drift' in result
        assert 'concept_drift' in result
        assert 'performance_drift' in result
        assert 'alerts' in result
        
        # Should detect drift in feature_1
        assert 'feature_1' in result['data_drift']
        
        # Check alerts
        assert len(result['alerts']) > 0
        for alert in result['alerts']:
            assert hasattr(alert, 'drift_type')
            assert hasattr(alert, 'severity')
            assert hasattr(alert, 'timestamp')
    
    def test_drift_summary(self):
        """Test drift summary functionality."""
        # Add some monitoring history first
        for i in range(5):
            current_features = self.baseline_features + np.random.randn(*self.baseline_features.shape) * 0.1
            current_targets = self.baseline_targets + np.random.randn(len(self.baseline_targets)) * 0.1
            current_predictions = pd.Series(self.model.predict(current_features))
            
            self.drift_monitor.monitor_drift(current_features, current_targets, current_predictions)
        
        summary = self.drift_monitor.get_drift_summary(lookback_periods=3)
        
        assert 'lookback_periods' in summary
        assert 'total_periods' in summary
        assert 'alert_counts' in summary
        assert 'drift_frequency' in summary
        assert summary['lookback_periods'] == 3


class TestABTesting:
    """Test A/B testing functionality."""
    
    def setup_method(self):
        """Set up test data."""
        np.random.seed(42)
        
        # Create test data
        n_samples = 100
        n_features = 4
        
        self.test_data = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        
        # Add target column
        self.test_data['target'] = (
            self.test_data.sum(axis=1) + np.random.randn(n_samples) * 0.1
        )
        
        # Create two models with different performance
        self.model_a = LinearRegression()
        self.model_b = RandomForestRegressor(n_estimators=5, random_state=42)
        
        # Train models on subset of data
        train_data = self.test_data.iloc[:80]
        train_features = train_data.drop(columns=['target'])
        train_targets = train_data['target']
        
        self.model_a.fit(train_features, train_targets)
        self.model_b.fit(train_features, train_targets)
        
        self.ab_tester = ABTester(minimum_sample_size=10)
    
    def test_statistical_tests(self):
        """Test statistical test methods."""
        # Create two sets of scores with different means
        scores_a = np.random.randn(50) + 0.5  # Higher mean
        scores_b = np.random.randn(50)        # Lower mean
        
        # Test t-test
        t_result = StatisticalTest.t_test(scores_a, scores_b)
        assert 'p_value' in t_result
        assert 'confidence_interval' in t_result
        assert 'effect_size' in t_result
        assert t_result['test_type'] == 't_test'
        
        # Test Mann-Whitney
        mw_result = StatisticalTest.mann_whitney_test(scores_a, scores_b)
        assert 'p_value' in mw_result
        assert 'effect_size' in mw_result
        assert mw_result['test_type'] == 'mann_whitney'
        
        # Test bootstrap
        boot_result = StatisticalTest.bootstrap_test(scores_a, scores_b, n_bootstrap=100)
        assert 'p_value' in boot_result
        assert 'confidence_interval' in boot_result
        assert boot_result['test_type'] == 'bootstrap'
    
    def test_ab_test_lifecycle(self):
        """Test complete A/B test lifecycle."""
        # Start test with more data
        test_id = self.ab_tester.start_ab_test(
            test_id='test_1',
            model_a=self.model_a,
            model_b=self.model_b,
            model_a_name='LinearRegression',
            model_b_name='RandomForest',
            test_data=self.test_data.iloc[80:],
            target_column='target',
            metric_name='mse'
        )
        
        assert test_id == 'test_1'
        assert 'test_1' in self.ab_tester.list_active_tests()
        
        # Check test status
        status = self.ab_tester.get_test_status('test_1')
        assert status['test_id'] == 'test_1'
        assert status['status'] == 'running'
        assert status['model_a_name'] == 'LinearRegression'
        assert status['model_b_name'] == 'RandomForest'
        
        # Add enough test data to meet minimum sample size
        for i in range(3):  # Add multiple batches
            additional_data = pd.DataFrame(
                np.random.randn(15, 4),
                columns=[f'feature_{i}' for i in range(4)]
            )
            additional_data['target'] = additional_data.sum(axis=1) + np.random.randn(15) * 0.1
            
            self.ab_tester.add_test_data('test_1', additional_data, 'target')
        
        # Check that we have enough data now
        status = self.ab_tester.get_test_status('test_1')
        assert status['sample_size'] >= self.ab_tester.minimum_sample_size
        
        # Evaluate test
        result = self.ab_tester.evaluate_test('test_1')
        if result is not None:  # Should have enough data now
            assert result.test_id == 'test_1'
            assert result.model_a_name == 'LinearRegression'
            assert result.model_b_name == 'RandomForest'
            assert result.metric_name == 'mse'
            assert isinstance(result.p_value, float)
            assert isinstance(result.effect_size, float)
        
        # Stop test
        final_result = self.ab_tester.stop_test('test_1')
        assert final_result.test_id == 'test_1'
        assert 'test_1' not in self.ab_tester.list_active_tests()
        
        # Check completed tests
        completed = self.ab_tester.get_completed_tests()
        assert len(completed) == 1
        assert completed[0].test_id == 'test_1'
    
    def test_sample_size_calculation(self):
        """Test sample size requirement calculation."""
        required_size = self.ab_tester.calculate_sample_size_requirement(
            effect_size=0.5,
            power=0.8,
            significance_level=0.05
        )
        
        assert isinstance(required_size, int)
        assert required_size >= self.ab_tester.minimum_sample_size
    
    def test_error_handling(self):
        """Test error handling in A/B testing."""
        # Test starting duplicate test
        self.ab_tester.start_ab_test(
            test_id='duplicate_test',
            model_a=self.model_a,
            model_b=self.model_b,
            model_a_name='Model A',
            model_b_name='Model B',
            test_data=self.test_data,
            target_column='target'
        )
        
        with pytest.raises(ValidationError):
            self.ab_tester.start_ab_test(
                test_id='duplicate_test',
                model_a=self.model_a,
                model_b=self.model_b,
                model_a_name='Model A',
                model_b_name='Model B',
                test_data=self.test_data,
                target_column='target'
            )
        
        # Test accessing non-existent test
        with pytest.raises(ValidationError):
            self.ab_tester.get_test_status('non_existent_test')


if __name__ == '__main__':
    pytest.main([__file__])