"""
Model drift detection and performance degradation monitoring.

This module provides tools for detecting when ML models are experiencing
performance degradation due to data drift, concept drift, or other factors.
"""

from typing import Dict, Any, List, Optional, Tuple, Union, Callable
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

from src.domain.exceptions import ValidationError


class DriftType(Enum):
    """Types of drift that can be detected."""
    DATA_DRIFT = "data_drift"
    CONCEPT_DRIFT = "concept_drift"
    PERFORMANCE_DRIFT = "performance_drift"


@dataclass
class DriftAlert:
    """Alert for detected drift."""
    drift_type: DriftType
    severity: str  # 'low', 'medium', 'high'
    metric_name: str
    current_value: float
    baseline_value: float
    threshold: float
    timestamp: datetime
    description: str
    confidence: float = 0.0


class StatisticalDriftDetector:
    """Statistical tests for detecting data drift."""
    
    def __init__(
        self,
        significance_level: float = 0.05,
        min_samples: int = 30
    ):
        """
        Initialize statistical drift detector.
        
        Args:
            significance_level: Significance level for statistical tests
            min_samples: Minimum samples required for testing
        """
        self.significance_level = significance_level
        self.min_samples = min_samples
    
    def kolmogorov_smirnov_test(
        self,
        baseline_data: pd.Series,
        current_data: pd.Series
    ) -> Dict[str, Any]:
        """
        Perform Kolmogorov-Smirnov test for distribution drift.
        
        Args:
            baseline_data: Baseline data distribution
            current_data: Current data distribution
            
        Returns:
            Dictionary with test results
        """
        if len(baseline_data) < self.min_samples or len(current_data) < self.min_samples:
            return {
                'drift_detected': False,
                'p_value': 1.0,
                'statistic': 0.0,
                'error': 'Insufficient samples for testing'
            }
        
        try:
            statistic, p_value = stats.ks_2samp(baseline_data, current_data)
            drift_detected = p_value < self.significance_level
            
            return {
                'drift_detected': drift_detected,
                'p_value': p_value,
                'statistic': statistic,
                'test': 'kolmogorov_smirnov',
                'significance_level': self.significance_level
            }
        except Exception as e:
            return {
                'drift_detected': False,
                'p_value': 1.0,
                'statistic': 0.0,
                'error': str(e)
            }
    
    def mann_whitney_test(
        self,
        baseline_data: pd.Series,
        current_data: pd.Series
    ) -> Dict[str, Any]:
        """
        Perform Mann-Whitney U test for distribution shift.
        
        Args:
            baseline_data: Baseline data
            current_data: Current data
            
        Returns:
            Dictionary with test results
        """
        if len(baseline_data) < self.min_samples or len(current_data) < self.min_samples:
            return {
                'drift_detected': False,
                'p_value': 1.0,
                'statistic': 0.0,
                'error': 'Insufficient samples for testing'
            }
        
        try:
            statistic, p_value = stats.mannwhitneyu(
                baseline_data, current_data, alternative='two-sided'
            )
            drift_detected = p_value < self.significance_level
            
            return {
                'drift_detected': drift_detected,
                'p_value': p_value,
                'statistic': statistic,
                'test': 'mann_whitney',
                'significance_level': self.significance_level
            }
        except Exception as e:
            return {
                'drift_detected': False,
                'p_value': 1.0,
                'statistic': 0.0,
                'error': str(e)
            }
    
    def chi_square_test(
        self,
        baseline_data: pd.Series,
        current_data: pd.Series,
        bins: int = 10
    ) -> Dict[str, Any]:
        """
        Perform Chi-square test for categorical drift.
        
        Args:
            baseline_data: Baseline data
            current_data: Current data
            bins: Number of bins for continuous data
            
        Returns:
            Dictionary with test results
        """
        if len(baseline_data) < self.min_samples or len(current_data) < self.min_samples:
            return {
                'drift_detected': False,
                'p_value': 1.0,
                'statistic': 0.0,
                'error': 'Insufficient samples for testing'
            }
        
        try:
            # Create bins for continuous data
            combined_data = pd.concat([baseline_data, current_data])
            bin_edges = np.histogram_bin_edges(combined_data, bins=bins)
            
            # Calculate histograms
            baseline_hist, _ = np.histogram(baseline_data, bins=bin_edges)
            current_hist, _ = np.histogram(current_data, bins=bin_edges)
            
            # Avoid zero frequencies
            baseline_hist = baseline_hist + 1
            current_hist = current_hist + 1
            
            # Perform chi-square test
            statistic, p_value = stats.chisquare(current_hist, baseline_hist)
            drift_detected = p_value < self.significance_level
            
            return {
                'drift_detected': drift_detected,
                'p_value': p_value,
                'statistic': statistic,
                'test': 'chi_square',
                'bins': bins,
                'significance_level': self.significance_level
            }
        except Exception as e:
            return {
                'drift_detected': False,
                'p_value': 1.0,
                'statistic': 0.0,
                'error': str(e)
            }


class PerformanceDriftDetector:
    """Detector for model performance degradation."""
    
    def __init__(
        self,
        performance_threshold: float = 0.1,
        window_size: int = 100,
        min_periods: int = 30
    ):
        """
        Initialize performance drift detector.
        
        Args:
            performance_threshold: Threshold for performance degradation
            window_size: Size of rolling window for monitoring
            min_periods: Minimum periods required for calculation
        """
        self.performance_threshold = performance_threshold
        self.window_size = window_size
        self.min_periods = min_periods
    
    def detect_performance_drift(
        self,
        y_true: pd.Series,
        y_pred: pd.Series,
        baseline_performance: Dict[str, float],
        metrics: List[str] = ['mse', 'mae', 'correlation']
    ) -> Dict[str, Any]:
        """
        Detect performance drift by comparing current to baseline performance.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            baseline_performance: Baseline performance metrics
            metrics: List of metrics to monitor
            
        Returns:
            Dictionary with drift detection results
        """
        if len(y_true) != len(y_pred):
            raise ValidationError("y_true and y_pred must have the same length")
        
        if len(y_true) < self.min_periods:
            return {
                'drift_detected': False,
                'error': 'Insufficient data for performance monitoring'
            }
        
        # Calculate current performance
        current_performance = self._calculate_performance_metrics(y_true, y_pred, metrics)
        
        # Check for drift in each metric
        drift_results = {}
        overall_drift = False
        
        for metric in metrics:
            if metric not in baseline_performance:
                continue
            
            baseline_value = baseline_performance[metric]
            current_value = current_performance[metric]
            
            # Calculate relative change
            if baseline_value != 0:
                relative_change = abs(current_value - baseline_value) / abs(baseline_value)
            else:
                relative_change = abs(current_value)
            
            # Check if drift exceeds threshold
            drift_detected = relative_change > self.performance_threshold
            
            drift_results[metric] = {
                'drift_detected': drift_detected,
                'baseline_value': baseline_value,
                'current_value': current_value,
                'relative_change': relative_change,
                'threshold': self.performance_threshold
            }
            
            if drift_detected:
                overall_drift = True
        
        return {
            'drift_detected': overall_drift,
            'metrics': drift_results,
            'current_performance': current_performance,
            'baseline_performance': baseline_performance
        }
    
    def monitor_rolling_performance(
        self,
        y_true: pd.Series,
        y_pred: pd.Series,
        timestamps: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """
        Monitor performance using rolling windows.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            timestamps: Timestamps for data points
            
        Returns:
            DataFrame with rolling performance metrics
        """
        if len(y_true) != len(y_pred):
            raise ValidationError("y_true and y_pred must have the same length")
        
        if timestamps is None:
            timestamps = pd.Series(range(len(y_true)))
        
        # Create DataFrame
        data = pd.DataFrame({
            'y_true': y_true,
            'y_pred': y_pred,
            'timestamp': timestamps
        })
        
        # Calculate rolling metrics
        rolling_metrics = []
        
        for i in range(self.window_size, len(data) + 1):
            window_data = data.iloc[i - self.window_size:i]
            
            metrics = self._calculate_performance_metrics(
                window_data['y_true'],
                window_data['y_pred'],
                ['mse', 'mae', 'correlation']
            )
            
            metrics['timestamp'] = window_data['timestamp'].iloc[-1]
            metrics['window_end'] = i
            rolling_metrics.append(metrics)
        
        return pd.DataFrame(rolling_metrics)
    
    def _calculate_performance_metrics(
        self,
        y_true: pd.Series,
        y_pred: pd.Series,
        metrics: List[str]
    ) -> Dict[str, float]:
        """Calculate performance metrics."""
        results = {}
        
        if 'mse' in metrics:
            results['mse'] = mean_squared_error(y_true, y_pred)
        
        if 'mae' in metrics:
            results['mae'] = mean_absolute_error(y_true, y_pred)
        
        if 'rmse' in metrics:
            results['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
        
        if 'correlation' in metrics:
            correlation = y_true.corr(y_pred)
            results['correlation'] = correlation if not np.isnan(correlation) else 0.0
        
        return results


class ModelDriftMonitor:
    """Comprehensive model drift monitoring system."""
    
    def __init__(
        self,
        statistical_detector: Optional[StatisticalDriftDetector] = None,
        performance_detector: Optional[PerformanceDriftDetector] = None,
        alert_thresholds: Optional[Dict[str, float]] = None
    ):
        """
        Initialize model drift monitor.
        
        Args:
            statistical_detector: Statistical drift detector
            performance_detector: Performance drift detector
            alert_thresholds: Thresholds for different alert levels
        """
        self.statistical_detector = statistical_detector or StatisticalDriftDetector()
        self.performance_detector = performance_detector or PerformanceDriftDetector()
        
        self.alert_thresholds = alert_thresholds or {
            'low': 0.05,
            'medium': 0.15,
            'high': 0.30
        }
        
        self.baseline_data: Dict[str, Any] = {}
        self.monitoring_history: List[Dict[str, Any]] = []
    
    def set_baseline(
        self,
        features: pd.DataFrame,
        targets: pd.Series,
        predictions: pd.Series,
        performance_metrics: Dict[str, float]
    ) -> None:
        """
        Set baseline data for drift detection.
        
        Args:
            features: Baseline feature data
            targets: Baseline target data
            predictions: Baseline predictions
            performance_metrics: Baseline performance metrics
        """
        self.baseline_data = {
            'features': features.copy(),
            'targets': targets.copy(),
            'predictions': predictions.copy(),
            'performance_metrics': performance_metrics.copy(),
            'timestamp': datetime.now()
        }
    
    def monitor_drift(
        self,
        features: pd.DataFrame,
        targets: pd.Series,
        predictions: pd.Series,
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Monitor for various types of drift.
        
        Args:
            features: Current feature data
            targets: Current target data
            predictions: Current predictions
            feature_names: Names of features to monitor
            
        Returns:
            Dictionary with drift monitoring results
        """
        if not self.baseline_data:
            raise ValidationError("Baseline data must be set before monitoring")
        
        monitoring_results = {
            'timestamp': datetime.now(),
            'data_drift': {},
            'concept_drift': {},
            'performance_drift': {},
            'alerts': []
        }
        
        # Monitor data drift (feature distribution changes)
        if feature_names is None:
            feature_names = features.columns.tolist()
        
        for feature_name in feature_names:
            if feature_name in self.baseline_data['features'].columns:
                baseline_feature = self.baseline_data['features'][feature_name]
                current_feature = features[feature_name]
                
                # Perform statistical tests
                ks_result = self.statistical_detector.kolmogorov_smirnov_test(
                    baseline_feature, current_feature
                )
                
                monitoring_results['data_drift'][feature_name] = ks_result
                
                # Generate alerts if drift detected
                if ks_result['drift_detected']:
                    severity = self._determine_severity(ks_result['p_value'])
                    alert = DriftAlert(
                        drift_type=DriftType.DATA_DRIFT,
                        severity=severity,
                        metric_name=f"feature_{feature_name}",
                        current_value=ks_result['statistic'],
                        baseline_value=0.0,
                        threshold=self.statistical_detector.significance_level,
                        timestamp=datetime.now(),
                        description=f"Data drift detected in feature {feature_name}",
                        confidence=1 - ks_result['p_value']
                    )
                    monitoring_results['alerts'].append(alert)
        
        # Monitor concept drift (target distribution changes)
        baseline_targets = self.baseline_data['targets']
        concept_drift_result = self.statistical_detector.kolmogorov_smirnov_test(
            baseline_targets, targets
        )
        monitoring_results['concept_drift'] = concept_drift_result
        
        if concept_drift_result['drift_detected']:
            severity = self._determine_severity(concept_drift_result['p_value'])
            alert = DriftAlert(
                drift_type=DriftType.CONCEPT_DRIFT,
                severity=severity,
                metric_name="target_distribution",
                current_value=concept_drift_result['statistic'],
                baseline_value=0.0,
                threshold=self.statistical_detector.significance_level,
                timestamp=datetime.now(),
                description="Concept drift detected in target distribution",
                confidence=1 - concept_drift_result['p_value']
            )
            monitoring_results['alerts'].append(alert)
        
        # Monitor performance drift
        baseline_performance = self.baseline_data['performance_metrics']
        performance_drift_result = self.performance_detector.detect_performance_drift(
            targets, predictions, baseline_performance
        )
        monitoring_results['performance_drift'] = performance_drift_result
        
        if performance_drift_result['drift_detected']:
            for metric_name, metric_result in performance_drift_result['metrics'].items():
                if metric_result['drift_detected']:
                    severity = self._determine_performance_severity(
                        metric_result['relative_change']
                    )
                    alert = DriftAlert(
                        drift_type=DriftType.PERFORMANCE_DRIFT,
                        severity=severity,
                        metric_name=metric_name,
                        current_value=metric_result['current_value'],
                        baseline_value=metric_result['baseline_value'],
                        threshold=metric_result['threshold'],
                        timestamp=datetime.now(),
                        description=f"Performance drift detected in {metric_name}",
                        confidence=metric_result['relative_change']
                    )
                    monitoring_results['alerts'].append(alert)
        
        # Store monitoring history
        self.monitoring_history.append(monitoring_results)
        
        return monitoring_results
    
    def get_drift_summary(self, lookback_periods: int = 10) -> Dict[str, Any]:
        """
        Get summary of drift detection over recent periods.
        
        Args:
            lookback_periods: Number of recent periods to summarize
            
        Returns:
            Dictionary with drift summary
        """
        if not self.monitoring_history:
            return {'error': 'No monitoring history available'}
        
        recent_history = self.monitoring_history[-lookback_periods:]
        
        # Count alerts by type and severity
        alert_counts = {
            'data_drift': {'low': 0, 'medium': 0, 'high': 0},
            'concept_drift': {'low': 0, 'medium': 0, 'high': 0},
            'performance_drift': {'low': 0, 'medium': 0, 'high': 0}
        }
        
        for period in recent_history:
            for alert in period['alerts']:
                drift_type = alert.drift_type.value
                severity = alert.severity
                alert_counts[drift_type][severity] += 1
        
        # Calculate drift frequency
        total_periods = len(recent_history)
        drift_frequency = {
            'data_drift': sum(alert_counts['data_drift'].values()) / total_periods,
            'concept_drift': sum(alert_counts['concept_drift'].values()) / total_periods,
            'performance_drift': sum(alert_counts['performance_drift'].values()) / total_periods
        }
        
        return {
            'lookback_periods': lookback_periods,
            'total_periods': total_periods,
            'alert_counts': alert_counts,
            'drift_frequency': drift_frequency,
            'last_monitoring': recent_history[-1]['timestamp'] if recent_history else None
        }
    
    def _determine_severity(self, p_value: float) -> str:
        """Determine alert severity based on p-value."""
        if p_value < 0.001:
            return 'high'
        elif p_value < 0.01:
            return 'medium'
        else:
            return 'low'
    
    def _determine_performance_severity(self, relative_change: float) -> str:
        """Determine alert severity based on performance change."""
        if relative_change >= self.alert_thresholds['high']:
            return 'high'
        elif relative_change >= self.alert_thresholds['medium']:
            return 'medium'
        else:
            return 'low'